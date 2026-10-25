# InstinctMJ SuperMimic parkour

Native task: `Instinct-Parkour-Target-Amp-G1-v0`. Reference algorithm:
[zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed).
This experimental extension is not an exact replication of SuperMimic or a
controlled comparison with the original PPO/AMP trainer.

## Paths on ai2-root-4080

- Code: `/nix/ykj/instinct_supermimic/InstinctMJ`
- Dedicated uv environment: `/nix/ykj/instinct_supermimic/.venv`
- Build/intermediate files and caches: `/nix/ykj/instinct_supermimic/{build,cache}`
- Input assets: `/workspace/ykj/instinct_supermimic/assets`
- Logs/checkpoints: `/workspace/ykj/instinct_supermimic/logs/<date_run>`

The server has A800 GPUs despite its SSH alias. Drivers/system CUDA are untouched.
Python 3.12, PyTorch 2.11 cu126, mjlab 1.3, MuJoCo 3.7 and MuJoCo-Warp 3.7.0.1
are isolated in the environment. `requirements-resolved.txt` records the tested
installation; GPU runtime packages are environment-local.

## Algorithm

The simulator retains every native parkour reward term, termination condition,
terrain category, target velocity command, domain randomization and curriculum.
The default terrain grid uses 5 difficulty rows and 20 columns (all ten terrain
categories), instead of the upstream ten rows. No changes to `instinct_rl`.

Training alternates real simulation collection, eight-step deterministic world
model fitting, a least-squares AMP discriminator update, 32-step policy BPTT
through the frozen world model and five GAE(lambda) critic updates. The policy
uses a bounded 32-dimensional goal encoder and ten proprioceptive frames, as in
the reference implementation. The velocity command and pooled latest depth image
replace future-motion goals. Collection adds Gaussian exploration noise (0.1).

The world model predicts normalized feature increments, native reward terms and
termination probabilities. Inputs include contacts, depth, local root position,
applied torque and previous action. Simple velocity/action/orientation rewards
are evaluated analytically during imagination; remaining sensor-dependent terms
are predicted from original reward-manager labels. These are material algorithm
changes required for this initial parkour experiment. The policy is not the
original PPO CNN/MoE policy and this initial depth input is pooled to 9x16.

The original parkour AMP reference distribution is not bundled. The initial
library combines the user's retargeted contact motions (knee climb, jump-sit,
vault and dive rolls) with all 40 public LAFAN1 G1 clips already on this server.
CSV conversion records explicit joint names and converts xyzw to wxyz; InstinctMJ
then reorders by name. `manifest.json` records hashes and source paths. This is
not a claim to reproduce the original parkour motion selection.

Replay masks exclude reset transitions from state regression and stop multistep
regression at episode boundaries. Native reward labels are checked against the
sum of individual terms on every collection step. Nonfinite loss/gradient/state
checks stop before optimizer updates. AMP retains quad reward with coefficient
0.25. Critic bootstraps are capped using a task reward upper bound, and raw
critic regression can correct overestimated values. Imagined and real actions
share the same [-20,20] execution bound. These stability measures should be
included in any reported comparison.

## Data preparation

Each command below is a single line; run them in order.

```bash
source /nix/ykj/instinct_supermimic/InstinctMJ/tools/supermimic/env.sh
```

```bash
python tools/supermimic/prepare_data.py --source "$INSTINCT_SM_DATA/20251116_50cm_kneeClimbStep1" --lafan /workspace/ykj/supermimic_vibed/code/data/lafan1_g1 --output "$INSTINCT_SM_DATA/parkour_amp"
```

## Training

Each command below is a single line. The default run is a fresh 12-hour single
GPU experiment, with 1024 environments, 512-step replay, 500 world-model-only
warmup updates, then policy/critic updates. This launcher acquires a task-specific
lock; a second trainer cannot accidentally overlap it. Keep it under a persistent
execution/session manager when disconnecting.

```bash
bash /nix/ykj/instinct_supermimic/InstinctMJ/tools/supermimic/run_parkour.sh 2026-10-25_supermimic_parkour_seed0
```

Short startup validation (also exercises BPTT and critic, not just environment creation):

```bash
bash /nix/ykj/instinct_supermimic/InstinctMJ/tools/supermimic/run_parkour.sh 2026-10-25_startup_validation --iterations 3 --num-envs 64 --terrain-rows 2 --terrain-cols 20 --wm-batch 32 --pi-batch 32 --wm-batches 1 --wm-only-iters 0 --max-hours 0 --log-every 1
```

Resume uses an explicit checkpoint from `last.json`. Checkpoints include all four
networks, normalizers and optimizer states. Replay is recollected on resume;
resume is not bitwise identical to uninterrupted training. `--max-hours` is the
budget for the new invocation. The launcher never overwrites existing checkpoints
unless explicitly resumed in the same directory with matching iteration names.

`train.csv` reports **real simulator** rewards, termination frequency, completed
episode duration/return, contact body count and curriculum level alongside model,
discriminator, policy and critic losses. Reward is not normalized to LAFAN's
flat-ground metric. Successful updates/checkpoints establish a working training
pipeline; they do not establish parkour success or multi-contact generalization.
