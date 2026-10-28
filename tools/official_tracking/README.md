# Official SuperMimic motion tracking reproduction

Upstream: project-instinct/InstinctMJ `supermimic`, commit
`7ca851e002980011f24a1664011305d0ba94aefc` (verified 2026-10-28).
Task: `Instinct-Perceptive-Shadowing-G1-v0`, not locomotion.

The launch follows the upstream **reported-results** settings: 1024 environments,
8000 iterations, `fail_head=False`, `wm_failure_frac=0`, `wm_holdout_envs=0`.
Seed 42 and all other learning/task settings are upstream defaults: 500 model-only
iterations, 8-step world-model windows, 10 batches of 2048, 32-step imagination,
policy batch 1024, 32-D visual latent, original rewards/terminations/noise/domain
randomization/adaptive motion sampling. Original save interval 2000 and log
interval 50 are preserved. No checkpoint is loaded for production training.

Dataset: fresh official `supermimic-v0.1` release archive, SHA256
`9d84a5efb5f08d3e243326f16ffcfc93c2721db50b38a0bbef99354ca2f93418`.
Data and heightmaps stay under `/workspace/ykj/instinct_supermimic/assets/`.
Logs: `/workspace/ykj/instinct_supermimic/logs/tracking_official`.
Code/environment: `/nix/ykj/instinct_tracking_official/{InstinctMJ,.venv}`.
Dependencies come from the unchanged official uv.lock, with mjlab v1.6.0 at
`../mjlab`. No changes to CUDA/GPU drivers or other environments.

Server-only setup and commands (each command is one line):

```bash
cd /nix/ykj/instinct_tracking_official/InstinctMJ
source tools/official_tracking/env.sh
uv sync --frozen --python /root/miniforge/bin/python3.12
bash tools/official_tracking/run.sh check
bash tools/official_tracking/run.sh train 2026-10-28_official_tracking_validation --validation
bash tools/official_tracking/run.sh train 2026-10-28_official_tracking_seed42
bash tools/official_tracking/run.sh eval --checkpoint CHECKPOINT
```

Default physical GPU 0, overridable by TRACKING_GPU. Evaluation uses the official
evaluator (128 environments, motion-by-motion assignment, camera noise on,
tracking failure terminations restored, success means full motion completion).
It records scalar/JSON results without videos. Validation uses the full training
network and batch sizes, reducing only total/warmup iterations and log/save
intervals; its weights are never used to initialize production.

The only core-file adaptations are filesystem/logging operations: sequential
heightmap archive I/O for bosfs, status.json, disabled image logging and retention
of the last two checkpoint pairs plus best by training nonfailure/reward proxy.
That proxy is not an independent success rate. No learning update, model, reward,
termination, observation or sampling implementation is changed.

## Validation and launch (2026-10-28)

- Frozen environment: 157 compatible packages; torch 2.10.0, mjlab 1.6.0,
  mujoco/mujoco-warp 3.11.0, warp-lang 1.14.0, instinct-rl 1.0.2.
- Release archive hash verified. Its actual metadata has 10 motions and 6
  terrains (the upstream results prose says 5); the released metadata is intact.
- Official analytic-head test with its default seed 0: ALL CHECKS PASSED,
  63 terminal transitions. The extra seed-42 test has one failure: stale-state
  total reward max error 0.05258 versus 0.05 tolerance, mean 0.000853 versus
  0.002 tolerance. All other checks pass. This documented substep-timing
  approximation is retained; neither the reward nor test tolerance was changed.
- Full-size four-iteration validation, including two policy updates, exited 0.
  Final WM loss 2.06808, policy loss -0.818938, gradient norm 1.53918, value loss
  0.0245884; peak allocated memory 23.45 GiB. Checkpoint/sidecar reload and finite
  parameter checks passed. The official evaluator ran 128 episodes across all
  ten motions on the validation checkpoint; this only verifies the evaluation
  pipeline and is not a trained-performance result.
- AST comparison: all runner methods except logging in learn() and storage in
  save() match upstream. Official uv.lock and task/agent definitions are byte
  identical. Saved production agent.yaml differs only in the three published
  result flags, 8000 max iterations, run name and heightmap path.

Production run `2026-10-28_official_tracking_seed42` started 17:45:19 +08:00,
physical GPU 0, PID 3302463, code commit c0181ab. It starts from scratch and
restores the official 500 model-only iterations and 2000-iteration save interval.
Official evaluation watch: physical GPU 2, PID 3303015, 128 environments,
polling every 120 s, exits after 3600 s without a new checkpoint. Results go to
the run's `eval/eval_<iteration>.json` and `eval/results.csv`; no videos.
Both production processes use nohup with stdin from /dev/null. PID files and
launcher logs are in the log root; live training metrics are in run/status.json.
