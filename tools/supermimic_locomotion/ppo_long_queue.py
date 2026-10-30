"""Longer PPO convergence control; resume2086 into a separate run without changing its MDP."""

from datetime import datetime
import fcntl
import os
import signal
import subprocess
import sys
import time

from experiment_queue import LOGS, ROOT, RUNTIME, acquire_gpu, fingerprint, now, read_json, stage, write_json

FOLDER = LOGS / "ppo_long_20261031"
JOBS = [
    ("walk_amp", "Instinct-Parkour-Target-Amp-MatchedPPO-G1-v0"),
]


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "ppo_long_20261031.lock").open("a") as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = FOLDER / "manifest.json"
        if not manifest_path.exists():
            write_json(manifest_path, dict(source_sha256=fingerprint(), created_at=now(),
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                iterations=8000, num_envs=1024, seed=42, real_transitions=196608000,
                comparison="Continue legacy PPO walk model2086 to8000 in a new run; extra-sample convergence control, NOT sample-matched.",
                control="Official PPO/WasabiPPO hyperparameters; exact existing SM environment, including smooth dont_wait. Walk selection has 12 motions. This is not an unmodified official-task replication.",
                termination="torso >1N, tilt >1rad, root height <0.5m; time limit/boundary truncated; original AMP task additionally retains dataset_exhausted",
                queue="Wait for both original PPO controls and all evaluations, then GPU3. Preserve original2086 checkpoint."))
            write_json(FOLDER / "state.json", dict(status="queued", jobs=[dict(id=i, task=t, status="pending") for i, t in JOBS]))
        manifest, state = read_json(manifest_path), read_json(FOLDER / "state.json")
        dependency = LOGS / "ppo_controls_20261031" / "state.json"
        state.update(status="waiting_for_dependency", dependency=str(dependency))
        write_json(FOLDER / "state.json", state)
        while True:
            try:
                parent = read_json(dependency)
            except (OSError, ValueError):
                time.sleep(30)
                continue
            if parent.get("status") == "completed":
                break
            if parent.get("status") == "failed":
                state.update(status="failed", error="Original PPO queue failed")
                write_json(FOLDER / "state.json", state)
                raise RuntimeError(state["error"])
            time.sleep(30)
        for job in state["jobs"]:
            if job["status"] == "completed":
                continue
            if manifest["source_sha256"] != fingerprint():
                raise RuntimeError("Source changed after preparation")
            state.update(status="waiting_for_gpu", current=job["id"])
            write_json(FOLDER / "state.json", state)
            lock = None
            try:
                acquired = acquire_gpu(dict(gpu_pool=[3], max_gpu_used_mb=2048, max_gpu_utilization=35))
                while acquired is None:
                    time.sleep(20)
                    acquired = acquire_gpu(dict(gpu_pool=[3], max_gpu_used_mb=2048, max_gpu_utilization=35))
                gpu, uuid, lock = acquired
                job.setdefault("run_name", f"{datetime.now():%Y-%m-%d}_ppo_long_{job['id']}_seed42")
                job.update(status="training", gpu=gpu, gpu_uuid=uuid, started_at=now())
                state.update(status="running")
                write_json(FOLDER / "state.json", state)
                run = LOGS / job["run_name"]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT / "src"),
                    INSTINCT_PARKOUR_SELECTION=str(ROOT / "tools/supermimic_locomotion/selections/walk.yaml"))
                target = run / "model_8000.pt"
                if not target.exists():
                    command = [sys.executable, "-u", "-m", "instinct_mj.scripts.supermimic.train_parkour",
                        "--task", job["task"], "--num-envs", "1024", "--device", "cuda:0", "--log-dir", str(run),
                        "--agent.seed", "42", "--agent.run-name", job["run_name"], "--agent.log-interval", "50"]
                    initial_run = LOGS / "2026-10-31_ppo_control_walk_amp_seed42"
                    checkpoint_run, checkpoint_name, iteration = initial_run, "model_2086.pt", 2086
                    remaining = 8000 - iteration
                    if run.exists() and any(run.iterdir()):
                        saved = read_json(run / "checkpoints.json")
                        remaining = 8000 - saved["iteration"]
                        checkpoint_run, checkpoint_name = run, saved["latest"]
                    if not (checkpoint_run / checkpoint_name).is_file():
                        raise FileNotFoundError(checkpoint_run / checkpoint_name)
                    command += ["--agent.load-run", str(checkpoint_run), "--agent.load-checkpoint", checkpoint_name]
                    if remaining <= 0:
                        raise RuntimeError("Missing target despite complete checkpoint manifest")
                    command += ["--agent.max-iterations", str(remaining)]
                    stage(command, env, FOLDER / "logs" / f"{job['id']}_train.log", lock)
                if not target.exists():
                    raise RuntimeError("Missing final PPO checkpoint")
                job.update(status="evaluating", evaluations=[])
                write_json(FOLDER / "state.json", state)
                # Match the historical SM evaluation's reference selection as well.
                env["INSTINCT_PARKOUR_SELECTION"] = str(LOGS.parents[1] / "assets/parkour_amp/selection.yaml")
                for intervention in ("intact", "shuffled"):
                    for seed in (1234, 2345, 3456):
                        out = FOLDER / "evaluations" / f"{job['id']}_{intervention}_{seed}.json"
                        if not out.exists():
                            stage([sys.executable, "-u", "-m", "instinct_mj.scripts.supermimic.evaluate_parkour",
                                "--task", job["task"], "--checkpoint", str(target), "--output", str(out),
                                "--num-envs", "128", "--seed", str(seed), "--seconds", "20",
                                "--perception-intervention", intervention], env, FOLDER / "logs" / f"{job['id']}_{intervention}_{seed}.log", lock)
                        result = read_json(out)
                        assert result["iteration"] == 8000 and result["episodes"] == 128
                        job["evaluations"].append(str(out))
                job.update(status="completed", finished_at=now())
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
