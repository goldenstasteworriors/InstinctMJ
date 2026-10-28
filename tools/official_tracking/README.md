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
