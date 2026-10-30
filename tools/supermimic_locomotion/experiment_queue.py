"""Persistent serial training/evaluation worker, launched by project Supervisor.

All subprocesses remain foreground children. No chat timers or notifications.
"""

import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import itertools
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RUNTIME = Path(os.environ.get("INSTINCT_LOCO_RUNTIME", "/nix/ykj/instinct_locomotion"))
LOGS = Path(os.environ.get("INSTINCT_LOCO_LOGS", "/workspace/ykj/instinct_supermimic/logs/locomotion_official"))
DATA = Path(os.environ.get("INSTINCT_PARKOUR_MOTION_DIR", "/workspace/ykj/instinct_supermimic/assets/parkour_amp"))
TASK = "Instinct-Parkour-Target-Amp-SuperMimic-G1-v0"


def now():
    return datetime.now().astimezone().isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".next")
    staging.write_text(json.dumps(value, indent=2, allow_nan=False))
    staging.replace(path)


def fingerprint():
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted(BASE.glob("*.py")) + sorted(BASE.glob("*.sh"))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(plan, folder):
    if (folder / "manifest.json").exists():
        manifest = read_json(folder / "manifest.json")
        if manifest["plan"] != plan or manifest["source_sha256"] != fingerprint():
            raise RuntimeError("Existing queue plan/source differs; refusing to overwrite its manifest")
        return manifest
    ids = [j["id"] for j in plan["jobs"]]
    if len(ids) != len(set(ids)) or any(j["amp"] not in ("walk", "run", "none") for j in plan["jobs"]):
        raise ValueError("Invalid queue jobs")
    if any(not 0 <= j.get("critic_real_weight", 0.0) <= 1 for j in plan["jobs"]):
        raise ValueError("Invalid real critic weight")
    dependency = plan.get("after_queue")
    if dependency is not None and (not dependency or Path(dependency).name != dependency or dependency in (".", "..", plan["id"])):
        raise ValueError("Dependency must name a queue under the project log root")
    inventory = {}
    selection_root = DATA / plan["id"]
    selection_root.mkdir(parents=True, exist_ok=True)
    for kind in ("walk", "run"):
        source = BASE / "selections" / (kind + ".yaml")
        files = yaml.safe_load(source.read_text())["selected_files"]
        rows = []
        for rel in files:
            path = DATA / rel
            if not path.resolve().is_relative_to(DATA.resolve()):
                raise ValueError("Reference outside authorized dataset")
            with np.load(path) as motion:
                rows.append({"file": rel, "frames": len(motion["joint_pos"]),
                             "fps": float(motion["framerate"]), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        target = selection_root / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise FileExistsError(target)
        target.write_bytes(source.read_bytes())
        inventory[kind] = {"selection": str(target), "files": rows,
                           "seconds": sum(x["frames"] / x["fps"] for x in rows)}
    manifest = {"plan": plan, "created_at": now(), "source_sha256": fingerprint(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "inventory": inventory,
                "notes": ["All production jobs start fresh, seed 42, smooth dont_wait, unchanged task/WM inputs.",
                          "No AMP: style_reward_coef=0 and style_minibatches=0. The unused discriminator and reference plumbing remain for identical model structure; no style reward/gradient or discriminator updates.",
                          "Run pool includes four run and two sprint clips. No-AMP loader retains original selection, unused by the policy.",
                          "Evaluation always uses the original reference selection for common environment initialization; references are not actor inputs.",
                          "Real failures: torso contact >1N, tilt >1rad, root height <0.5m. Timeout/boundary are truncations. Evaluation uses pre-reset first episodes, no curriculum updates.",
                          "WM inputs remain state/action, depth latent, privileged terrain scan, delays/friction/previous action. Predicted contacts and analytic termination remain unchanged.",
                          "fail_head=False disables use in imagination, not failure-head fitting. holdout=0 is not held-out validation.",
                          "No scheduled chat notifications. No videos. Latest two checkpoint pairs plus training-proxy best retained."]}
    if "notes" in plan:
        manifest["notes"] = plan["notes"]
    write_json(folder / "manifest.json", manifest)
    write_json(folder / "state.json", {"status": "queued", "created_at": now(), "jobs": [dict(j, status="pending") for j in plan["jobs"]]})
    print(json.dumps({"queue": str(folder), "jobs": ids, "inventory": {k: len(v["files"]) for k, v in inventory.items()}}), flush=True)
    return manifest


def training_command(plan, job, run_dir, checkpoint=None):
    flags = job["flags"]
    cmd = [sys.executable, "-u", "-m", "instinct_mj.scripts.supermimic.train_parkour",
           "--task", plan.get("task", TASK), "--num-envs", str(plan["num_envs"]), "--device", "cuda:0", "--log-dir", str(run_dir),
           "--agent.seed", str(plan["seed"]), "--agent.run-name", run_dir.name,
           "--agent.max-iterations", str(plan["iterations"]), "--agent.save-interval", "500",
           "--agent.heightmap-cache-dir", os.environ["INSTINCT_HEIGHTMAP_CACHE"],
           "--agent.fail-head", str(flags[0]), "--agent.wm-failure-frac", str(flags[1]),
           "--agent.wm-holdout-envs", str(flags[2]),
           "--agent.style-reward-coef", "0" if job["amp"] == "none" else "0.25",
           "--agent.style-minibatches", "0" if job["amp"] == "none" else "2"]
    if job.get("critic_real_weight", 0.0):
        cmd += ["--agent.critic-real-weight", str(job["critic_real_weight"])]
    if job.get("wm_stratified_holdout", False):
        cmd += ["--agent.wm-stratified-holdout", "True", "--agent.wm-holdout-seed", str(plan["seed"])]
    if checkpoint is not None:
        cmd += ["--agent.load-run", str(run_dir), "--agent.load-checkpoint", checkpoint]
    return cmd


def acquire_gpu(plan):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    for row in csv.reader(output.splitlines()):
        index, uuid, used, util = [x.strip() for x in row]
        if int(index) not in plan["gpu_pool"] or int(used) > plan["max_gpu_used_mb"] or int(util) > plan["max_gpu_utilization"]:
            continue
        lock = open(RUNTIME / f"train_gpu{index}.lock", "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return int(index), uuid, lock
        except BlockingIOError:
            lock.close()
    return None


def stage(command, env, logfile, lock):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(logfile, maxBytes=20 * 1024**2, backupCount=2)
    logger = logging.Logger(str(logfile), logging.INFO)
    logger.addHandler(handler)
    print(json.dumps({"started_at": now(), "command": command, "gpu": env["CUDA_VISIBLE_DEVICES"], "log": str(logfile)}), flush=True)
    process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace", bufsize=1, pass_fds=(lock.fileno(),))
    try:
        for line in process.stdout:
            logger.info(line.rstrip("\n"))
        code = process.wait()
        if code:
            raise RuntimeError(f"Child exited {code}; inspect {logfile}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()
        handler.close()


def aggregate(folder, state):
    rows = []
    for job in state["jobs"]:
        if job["status"] != "completed":
            continue
        groups = {}
        for path in job["evaluations"]:
            result = read_json(path)
            groups.setdefault(result.get("perception_intervention", "intact"), []).extend(result["records"])
        for intervention, records in groups.items():
            seconds = sum(r["seconds"] for r in records)
            rows.append({"job": job["id"], "run": job["run_name"], "perception_intervention": intervention, "episodes": len(records),
                         "survival_fraction": sum(r["result"] == "survived_horizon" for r in records) / len(records),
                         "mean_seconds": seconds / len(records),
                         "linear_velocity_rmse_mps": math.sqrt(sum(r["linear_velocity_rmse_mps"]**2 * r["seconds"] for r in records) / seconds),
                         "yaw_velocity_rmse_radps": math.sqrt(sum(r["yaw_velocity_rmse_radps"]**2 * r["seconds"] for r in records) / seconds),
                         "mean_travel_m": sum(r["travel_xy_m"] for r in records) / len(records),
                         "mean_displacement_m": sum(r["displacement_xy_m"] for r in records) / len(records)})
    write_json(folder / "comparison.json", rows)
    if rows:
        with (folder / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def work(plan, folder, manifest):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / (plan["id"] + ".lock")).open("a") as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(folder / "state.json")
        if plan.get("after_queue") and any(j["status"] != "completed" for j in state["jobs"]):
            dependency = LOGS / plan["after_queue"] / "state.json"
            state.update(status="waiting_for_dependency", dependency=str(dependency), updated_at=now())
            write_json(folder / "state.json", state)
            while True:
                try:
                    parent = read_json(dependency) if dependency.exists() else {}
                except (OSError, json.JSONDecodeError) as error:
                    # BOSFS can transiently reject reads of an actively replaced
                    # status file. Observation failure is not parent failure.
                    print(f"Dependency observation unavailable; retrying: {error}", flush=True)
                    time.sleep(30)
                    continue
                if parent.get("status") == "completed":
                    break
                if parent.get("status") == "failed":
                    state.update(status="failed", error="Dependency failed; no training started", updated_at=now())
                    write_json(folder / "state.json", state)
                    raise RuntimeError(state["error"])
                time.sleep(30)
        for job in state["jobs"]:
            if job["status"] == "completed":
                continue
            gpu_lock = None
            try:
                if fingerprint() != manifest["source_sha256"]:
                    raise RuntimeError("Source changed since queue preparation; stop for review")
                state.update(status="waiting_for_gpu", current=job["id"], updated_at=now())
                write_json(folder / "state.json", state)
                acquired = acquire_gpu(plan)
                while acquired is None:
                    time.sleep(30)
                    acquired = acquire_gpu(plan)
                gpu, uuid, gpu_lock = acquired
                if "run_name" not in job:
                    job["run_name"] = f"{datetime.now():%Y-%m-%d}_{plan['id']}_{job['id']}_seed{plan['seed']}"
                job.update(gpu=gpu, gpu_uuid=uuid, started_at=job.get("started_at", now()), status="training")
                state.update(status="running", updated_at=now())
                write_json(folder / "state.json", state)
                run_dir = LOGS / job["run_name"]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                env["INSTINCT_PARKOUR_SELECTION"] = str(DATA / "selection.yaml") if job["amp"] == "none" else manifest["inventory"][job["amp"]]["selection"]
                target = run_dir / f"model_{plan['iterations']}.pt"
                cp = None
                if not target.exists():
                    if run_dir.exists() and any(run_dir.iterdir()):
                        saved = read_json(run_dir / "checkpoints.json")
                        cp = saved["latest"]
                        if not (run_dir / cp).exists() or not (run_dir / cp.replace("model_", "supermimic_state_")).exists():
                            raise RuntimeError("Incomplete checkpoint pair; refusing to overwrite")
                    stage(training_command(plan, job, run_dir, cp), env, folder / "logs" / (job["id"] + "_train.log"), gpu_lock)
                status = read_json(run_dir / "status.json")
                if status["iteration"] != plan["iterations"] or not target.exists() or not target.with_name(target.name.replace("model_", "supermimic_state_")).exists():
                    raise RuntimeError("Training did not produce the requested final checkpoint pair")
                job.update(status="evaluating", checkpoint=str(target), evaluations=[])
                write_json(folder / "state.json", state)
                # Common reference plumbing gives every policy the same evaluation setup.
                env["INSTINCT_PARKOUR_SELECTION"] = str(DATA / "selection.yaml")
                for intervention, seed in itertools.product(plan.get("eval_interventions", ["intact"]), plan["eval_seeds"]):
                    output = folder / "evaluations" / f"{job['id']}_{intervention}_seed{seed}.json"
                    if not output.exists():
                        cmd = [sys.executable, "-u", "-m", "instinct_mj.scripts.supermimic.evaluate_parkour", "--task", plan.get("task", TASK), "--checkpoint", str(target),
                               "--output", str(output), "--num-envs", str(plan["eval_envs"]), "--seconds", str(plan["eval_seconds"]), "--seed", str(seed),
                               "--perception-intervention", intervention]
                        stage(cmd, env, folder / "logs" / f"{job['id']}_{intervention}_eval{seed}.log", gpu_lock)
                    result = read_json(output)
                    if result["iteration"] != plan["iterations"] or result["episodes"] != plan["eval_envs"] or result["seed"] != seed or result["checkpoint"] != str(target):
                        raise RuntimeError("Evaluation provenance mismatch")
                    job["evaluations"].append(str(output))
                    write_json(folder / "state.json", state)
                job.update(status="completed", finished_at=now())
                write_json(folder / "state.json", state)
                aggregate(folder, state)
            except BaseException as error:
                job.update(status="failed", error=repr(error), failed_at=now())
                state.update(status="failed", updated_at=now())
                write_json(folder / "state.json", state)
                raise
            finally:
                if gpu_lock is not None:
                    gpu_lock.close()
        state.update(status="completed", finished_at=now(), current=None)
        write_json(folder / "state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "status"))
    parser.add_argument("--plan", type=Path, default=BASE / "queue_20261030.json")
    args = parser.parse_args()
    plan = read_json(args.plan)
    folder = LOGS / plan["id"]
    if args.mode == "status":
        print(json.dumps(read_json(folder / "state.json"), indent=2))
        return
    manifest = prepare(plan, folder)
    if args.mode == "run":
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        work(plan, folder, manifest)


if __name__ == "__main__":
    main()
