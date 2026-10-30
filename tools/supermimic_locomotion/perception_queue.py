"""Run the two priority no-AMP perception experiments, then resume the old queue."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiment_queue import LOGS, ROOT, RUNTIME, acquire_gpu, fingerprint, now, read_json, stage, write_json

QUEUE = LOGS / "perception_noamp_20261030"
OLD = LOGS / "amp_single_ablation_20261030"
OLD_CONFIG = RUNTIME / "InstinctMJ/tools/supermimic_locomotion/supervisor_queue_20261030.conf"
GPU_PLAN = {"gpu_pool": [0, 3], "max_gpu_used_mb": 2048, "max_gpu_utilization": 35}


def prepare():
    manifest = QUEUE / "manifest.json"
    if manifest.exists():
        if read_json(manifest)["source_sha256"] != fingerprint():
            raise RuntimeError("Prepared source changed; refusing to mix implementations")
        return
    write_json(manifest, {
        "created_at": now(), "source_sha256": fingerprint(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "iterations": 20000, "num_envs": 1024, "seed": 42,
        "flags": {"fail_head": True, "wm_failure_frac": 0.25, "wm_holdout_envs": 0.1},
        "amp": "No reference sensor/observations/reset, no discriminator or style reward",
        "command_ranges": {"vx_mps": [0.2, 1.0], "vy_mps": [-0.3, 0.3], "yaw_radps": [-1.0, 1.0]},
        "wm_direct_terrain_scan": False,
        "privileged_information_remaining": "Simulator state, friction/delay context; geometry in analytic reward/critic/failure heads. Ideal height observation is privileged; depth comes from simulated camera.",
        "perception": {"depth": "8 consecutive 18x32 depth frames, 0..2.5m -> 0..1",
                       "height": "8 consecutive 11x17 yaw-aligned maps, 0.1m spacing, 1.6x1.0m; clip(root_z-ground_z-0.5,-1,1) -> 0..1"},
        "termination": "torso >1N, tilt >1rad, root_height <0.5m; 20s timeout and terrain boundary are truncations",
        "evaluation": {"seeds": [1234, 2345, 3456], "num_envs": 128, "seconds": 20,
                       "interventions": ["intact", "shuffled"], "note": "Survival is not obstacle completion. One training seed only."},
        "wm_diagnostics": "Every 1000 iterations, same held-out windows with intact/shuffled/mean latent; physical RMSE and contact BCE at 1/8/16/32 steps; RNG restored",
        "priority": "Wait until current old job completes; run depth then height; resume remaining old ablations",
    })
    write_json(QUEUE / "state.json", {"status": "queued", "created_at": now(), "jobs": [
        {"id": mode, "status": "pending", "task": f"Instinct-Velocity-NoAmp-{mode.title()}-SuperMimic-G1-v0"}
        for mode in ("depth", "height")
    ]})


def aggregate(state):
    rows = []
    for job in state["jobs"]:
        if job["status"] != "completed":
            continue
        for intervention in ("intact", "shuffled"):
            records = []
            for path in job["evaluations"]:
                result = read_json(path)
                if result["perception_intervention"] == intervention:
                    records += result["records"]
            seconds = sum(r["seconds"] for r in records)
            rows.append({"modality": job["id"], "intervention": intervention, "episodes": len(records),
                         "survival": sum(r["result"] == "survived_horizon" for r in records) / len(records),
                         "mean_seconds": seconds / len(records),
                         "linear_rmse_mps": (sum(r["linear_velocity_rmse_mps"]**2 * r["seconds"] for r in records) / seconds)**0.5,
                         "yaw_rmse_radps": (sum(r["yaw_velocity_rmse_radps"]**2 * r["seconds"] for r in records) / seconds)**0.5,
                         "mean_travel_m": sum(r["travel_xy_m"] for r in records) / len(records)})
    write_json(QUEUE / "comparison.json", rows)


def resume_original():
    state = read_json(OLD / "state.json")
    if state["status"] == "completed":
        return
    if state["status"] != "paused_for_perception":
        raise RuntimeError("Original queue is not at the expected priority pause")
    for job in state["jobs"]:
        if "fresh_name_suffix" in job:
            job["run_name"] = f"{datetime.now():%Y-%m-%d}_amp_single_ablation_20261030_{job['id']}_seed42{job.pop('fresh_name_suffix')}"
    state.update(status="queued", resumed_after_perception_at=now())
    write_json(OLD / "state.json", state)
    subprocess.run([str(RUNTIME / "service-env/bin/supervisorctl"), "-c", str(OLD_CONFIG), "start", "amp_ablation_queue"], check=True)


def work():
    prepare()
    with (RUNTIME / "perception_noamp_20261030.lock").open("a") as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(QUEUE / "state.json")
        if state["status"] == "completed":
            return
        state.update(status="waiting_for_original_job", updated_at=now())
        write_json(QUEUE / "state.json", state)
        while read_json(OLD / "state.json")["status"] not in ("paused_for_perception", "completed"):
            if read_json(OLD / "state.json")["status"] == "failed":
                raise RuntimeError("Original queue failed before the handoff")
            time.sleep(20)
        for job in state["jobs"]:
            if job["status"] == "completed":
                continue
            gpu_lock = None
            try:
                if read_json(QUEUE / "manifest.json")["source_sha256"] != fingerprint():
                    raise RuntimeError("Source changed since preparation")
                state.update(status="waiting_for_gpu", current=job["id"])
                write_json(QUEUE / "state.json", state)
                acquired = acquire_gpu(GPU_PLAN)
                while acquired is None:
                    time.sleep(20)
                    acquired = acquire_gpu(GPU_PLAN)
                gpu, uuid, gpu_lock = acquired
                job.setdefault("run_name", f"{datetime.now():%Y-%m-%d}_perception_noamp_{job['id']}_seed42")
                job.update(status="training", gpu=gpu, gpu_uuid=uuid)
                job.setdefault("started_at", now())
                state.update(status="running", updated_at=now())
                write_json(QUEUE / "state.json", state)
                run = LOGS / job["run_name"]
                target = run / "model_20000.pt"
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT / "src"))
                if not target.exists():
                    command = [sys.executable, "-u", "-m", "instinct_mj.scripts.supermimic.train_parkour",
                               "--task", job["task"], "--num-envs", "1024", "--device", "cuda:0", "--log-dir", str(run),
                               "--agent.seed", "42", "--agent.run-name", job["run_name"], "--agent.max-iterations", "20000",
                               "--agent.heightmap-cache-dir", os.environ["INSTINCT_HEIGHTMAP_CACHE"]]
                    if run.exists() and any(run.iterdir()):
                        checkpoint = read_json(run / "checkpoints.json")["latest"]
                        if not (run / checkpoint.replace("model_", "supermimic_state_")).exists():
                            raise RuntimeError("Incomplete resume pair")
                        command += ["--agent.load-run", str(run), "--agent.load-checkpoint", checkpoint]
                    stage(command, env, QUEUE / "logs" / f"{job['id']}_train.log", gpu_lock)
                if read_json(run / "status.json")["iteration"] != 20000 or not (run / "supermimic_state_20000.pt").exists():
                    raise RuntimeError("Missing final training state")
                job.update(status="evaluating", evaluations=[])
                write_json(QUEUE / "state.json", state)
                for intervention in ("intact", "shuffled"):
                    for seed in (1234, 2345, 3456):
                        output = QUEUE / "evaluations" / f"{job['id']}_{intervention}_{seed}.json"
                        if not output.exists():
                            command = [sys.executable, "-u", "-m", "instinct_mj.scripts.supermimic.evaluate_parkour",
                                       "--task", job["task"], "--checkpoint", str(target), "--output", str(output),
                                       "--num-envs", "128", "--seconds", "20", "--seed", str(seed),
                                       "--perception-intervention", intervention]
                            stage(command, env, QUEUE / "logs" / f"{job['id']}_{intervention}_{seed}.log", gpu_lock)
                        result = read_json(output)
                        assert result["iteration"] == 20000 and result["checkpoint"] == str(target)
                        assert result["task"] == job["task"] and result["perception_intervention"] == intervention
                        assert result["seed"] == seed and result["episodes"] == 128
                        job["evaluations"].append(str(output))
                        write_json(QUEUE / "state.json", state)
                job.update(status="completed", finished_at=now())
                write_json(QUEUE / "state.json", state)
                aggregate(state)
            except BaseException as exc:
                job.update(status="failed", error=repr(exc))
                state.update(status="failed", updated_at=now())
                write_json(QUEUE / "state.json", state)
                raise
            finally:
                if gpu_lock is not None:
                    gpu_lock.close()
        resume_original()
        state.update(status="completed", current=None, finished_at=now(), original_queue_resumed=True)
        write_json(QUEUE / "state.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "status"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "status":
        print(json.dumps(read_json(QUEUE / "state.json"), indent=2))
    else:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        work()
