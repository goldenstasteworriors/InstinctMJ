# ai2-root-4080 deployment — 2026-10-25

Session: mbrl, 2026-10-25, user request turn 1 (including parkour selection).
Branch: `feat/supermimic-multicontact-20261025`, pushed using local GitHub SSH
authentication; server fetches the public branch over HTTPS because its own
GitHub SSH identity has no access.

## Verified

- Dedicated uv environment under `/nix/ykj/instinct_supermimic/.venv`;
  exact 175-package installed set in `requirements-resolved.txt`.
- `uv pip check`: all packages compatible. Ruff checks of new training/data
  modules and tools pass.
- GPU replay ring-buffer chronology after wraparound and episode reset masking
  passed; checkpoint network/normalizer tensors were checked finite, with all
  four optimizers present.
- Native edge merge acceleration matched the original function exactly on 42
  random/adversarial cases. Temporary regression scripts removed locally and
  remotely after testing.
- Small full-pipeline validation: 64 environments, three model/discriminator/
  BPTT-policy/critic updates, checkpoint save/reload and 128-environment real
  evaluation. All completed successfully with finite metrics and nonzero policy
  gradients.
- Full-size validation: 1024 environments, 5x20 terrain grid, full 512 replay
  capacity, ten world-model updates per iteration, 512-sample model/policy
  batches, 8-step model rollout and 32-step policy BPTT. Three complete updates
  followed by checkpoint/reload and real evaluation exited 0. Final validation
  model loss 3.6692, policy gradient norm 0.17395. These are startup checks,
  not trained parkour quality or convergence results.

Validation logs are under `/workspace/ykj/instinct_supermimic/logs/`:
`2026-10-25_startup_validation_torsobase` and
`2026-10-25_fullsize_validation`. Diagnostic runs from earlier initialization
attempts are preserved separately.

## Production invocation

Run name: `2026-10-25_supermimic_parkour_seed0`.
Fresh seed 0, physical GPU 0, 1024 environments, 100 terrain tiles across ten
terrain categories, 52 motions. No initialization from diagnostic checkpoints.
The first 500 iterations fit the world model and AMP discriminator; subsequent
iterations train the policy/critic through imagination as well. Save a full
checkpoint at the warmup boundary, every 1000 iterations and at clean exit.
Budget: 12 hours measured after environment initialization, including replay
warmup, followed by a 128-environment, 1000-step evaluation. Finite iteration cap
320000. Run detached from the SSH connection with nohup; task-specific flock
prevents a second trainer from using the same launcher concurrently.

Run directory:

```text
/workspace/ykj/instinct_supermimic/logs/2026-10-25_supermimic_parkour_seed0
```

`launcher.pid` records the shell launcher; `stdout.log`/`train.csv` report actual
updates. `protocol.json` documents algorithm differences. `last.json` points to
the latest closed checkpoint, `completed.json` marks clean training completion
and `final_eval.json` marks evaluation completion. There is no W&B run and no
automatic chat notification.

Native simulator rewards/terminations are preserved. This experiment changes
the policy/world-model representation and AMP dataset, and uses learned
sensor-dependent rewards during imagination; it is not an exact SuperMimic
paper replication or a controlled PPO comparison. See README for details.
