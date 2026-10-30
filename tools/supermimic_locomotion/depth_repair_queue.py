"""Serial fixed-friction control then terrain-supervised depth training."""

from datetime import datetime
import fcntl
import os
import signal
import subprocess
import sys
import time

from experiment_queue import LOGS, ROOT, RUNTIME, acquire_gpu, fingerprint, now, read_json, stage, write_json

FOLDER = LOGS / "depth_repair_20261031"
TASK = "Instinct-Velocity-NoAmp-Depth-SuperMimic-G1-v0"


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "depth_repair_20261031.lock").open("a") as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = FOLDER / "manifest.json"
        if not manifest_path.exists():
            write_json(manifest_path, dict(source_sha256=fingerprint(), created_at=now(),
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                iterations=20000, num_envs=1024, seed=42,
                correction="Declare geom_friction for per-world material randomization; verified 64 unique coefficients in 64 envs, std 0.3808.",
                ablation="Both jobs have corrected friction. Only perception_geometry_weight differs (0 vs 1).",
                perception="Same 8 consecutive depth frames. Geometry decoder supervises encoder from same-time yaw-aligned root-to-ground height labels, clipped 0..2.5m. Geometry labels are privileged training information, never policy/WM inputs.",
                termination="Unchanged: torso contact >1N, tilt >1rad, root height <0.5m; 20s timeout and terrain boundary truncated.",
                queue="Sequential on GPU2; existing PPO and original ablation queues untouched."))
            write_json(FOLDER / "state.json", dict(status="queued", jobs=[
                dict(id="friction_only", geometry_weight=0.0, status="pending"),
                dict(id="friction_geometry", geometry_weight=1.0, status="pending")]))
        manifest, state = read_json(manifest_path), read_json(FOLDER / "state.json")
        for job in state["jobs"]:
            if job["status"] == "completed":
                continue
            if manifest["source_sha256"] != fingerprint():
                raise RuntimeError("Source changed after preparation")
            state.update(status="waiting_for_gpu", current=job["id"])
            write_json(FOLDER / "state.json", state)
            lock = None
            try:
                plan = dict(gpu_pool=[2], max_gpu_used_mb=2048, max_gpu_utilization=35)
                acquired = acquire_gpu(plan)
                while acquired is None:
                    time.sleep(20)
                    acquired = acquire_gpu(plan)
                gpu, uuid, lock = acquired
                job.setdefault("run_name", f"{datetime.now():%Y-%m-%d}_depth_repair_{job['id']}_seed42")
                job.update(status="training", gpu=gpu, gpu_uuid=uuid, started_at=now())
                state.update(status="running")
                write_json(FOLDER / "state.json", state)
                run = LOGS / job["run_name"]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT / "src"))
                target = run / "model_20000.pt"
                if not target.exists():
                    command = [sys.executable, "-u", "-m", "instinct_mj.scripts.supermimic.train_parkour",
                        "--task", TASK, "--num-envs", "1024", "--device", "cuda:0", "--log-dir", str(run),
                        "--agent.seed", "42", "--agent.run-name", job["run_name"], "--agent.max-iterations", "20000",
                        "--agent.perception-geometry-weight", str(job["geometry_weight"]),
                        "--agent.heightmap-cache-dir", os.environ["INSTINCT_HEIGHTMAP_CACHE"]]
                    if run.exists() and any(run.iterdir()):
                        saved = read_json(run / "checkpoints.json")
                        command += ["--agent.load-run", str(run), "--agent.load-checkpoint", saved["latest"]]
                    stage(command, env, FOLDER / "logs" / f"{job['id']}_train.log", lock)
                if not target.exists() or not (run / "supermimic_state_20000.pt").exists():
                    raise RuntimeError("Missing complete final checkpoint pair")
                job.update(status="evaluating", evaluations=[])
                write_json(FOLDER / "state.json", state)
                for intervention in ("intact", "shuffled"):
                    for seed in (1234, 2345, 3456):
                        out = FOLDER / "evaluations" / f"{job['id']}_{intervention}_{seed}.json"
                        if not out.exists():
                            stage([sys.executable, "-u", "-m", "instinct_mj.scripts.supermimic.evaluate_parkour",
                                "--task", TASK, "--checkpoint", str(target), "--output", str(out),
                                "--num-envs", "128", "--seed", str(seed), "--seconds", "20",
                                "--perception-intervention", intervention], env, FOLDER / "logs" / f"{job['id']}_{intervention}_{seed}.log", lock)
                        result = read_json(out)
                        assert result["iteration"] == 20000 and result["episodes"] == 128
                        job["evaluations"].append(str(out))
                out = FOLDER / "evaluations" / f"{job['id']}_closed_loop.json"
                if not out.exists():
                    stage([sys.executable, "-u", "-m", "instinct_mj.scripts.supermimic.audit_closed_loop",
                        "--task", TASK, "--checkpoint", str(target), "--output", str(out),
                        "--num-envs", "128", "--windows", "16"], env, FOLDER / "logs" / f"{job['id']}_closed_loop.log", lock)
                job.update(status="completed", finished_at=now(), closed_loop=str(out))
                write_json(FOLDER / "state.json", state)
            except BaseException as exc:
                state.update(status="failed", error=repr(exc), updated_at=now())
                write_json(FOLDER / "state.json", state)
                raise
            finally:
                if lock is not None:
                    lock.close()
        state.update(status="completed", finished_at=now())
        write_json(FOLDER / "state.json", state)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
