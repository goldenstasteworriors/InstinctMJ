# No-AMP perception comparison — 2026-10-30

Two fresh SuperMimic velocity-locomotion runs, depth then elevation, each 20,000
iterations, 1,024 environments, seed 42. Both use failure head=True, failure
sampling=0.25, holdout=0.1. This is a matched modality comparison, not the previous
three-switch ablation. The old queue resumes after both experiments and evaluations.

## Task and complete AMP removal

Tasks are `Instinct-Velocity-NoAmp-Depth-SuperMimic-G1-v0` and
`Instinct-Velocity-NoAmp-Height-SuperMimic-G1-v0`.
No motion-reference sensor, AMP observation groups, dataset-exhaustion reset,
discriminator, discriminator optimizer, adversarial reward, or reference loading.
The inherited arithmetic AMP-history helpers are unused by the objective.

Commands are independently sampled body-frame velocities on all terrains:
vx=0.2..1.0 m/s, vy=-0.3..0.3 m/s, yaw=-1..1 rad/s, resampled every 8..12 s.
The old target-position-dependent velocity command is bypassed; heading-error
penalty is zero. Existing velocity-tracking rewards, smooth dont_wait, alive,
posture, effort, action-rate and contact penalties remain in both variants.
The terrain family and tracking-based curriculum remain unchanged.

Actual failure conditions remain torso contact >1 N, tilt >1 rad, root height
<0.5 m. Timeout at 20 s and leaving the terrain are truncations. Both training
and evaluation use these same conditions. Replay reads s1 and its perception
before reset; masks prevent world-model losses from crossing resets. Imagined
termination uses the existing smooth contact/tilt/height heads plus failure head.
No object or contact-loss termination is invented for this upright locomotion task.

## Perception and information availability

Both policy and world model use an encoder with latent dimension 128. The world
model consumes the latent and predicts its successor; imagined policy observations
use the predicted latent. Encoder training uses reconstruction; dynamics and latent
prediction targets use detached encoder outputs, following the existing method.

- Depth: processed simulated camera frames, 18x32, range 0..2.5 m encoded as 0..1.
- Elevation: 11x17 samples at 0.1 m spacing over 1.6x1.0 m, yaw-aligned at the
  pelvis. Values are clip(root_z - ground_z - 0.5, -1, 1), mapped to 0..1.
  Grid is y-major, x varies fastest. Ray sensor includes terrain geometry group 0.
  This is **ideal simulation privileged elevation**, not reconstruction from depth
  or a hardware-ready estimator. Misses saturate high; a heightmap cannot express
  overhangs. No artificial elevation noise is added.
- Both have eight consecutive observation-manager frames (50 Hz), oldest first,
  reset through the same observation history manager, with no extra frame delay.
  This aligns temporal support instead of preserving the old depth subsampling.
- Replay stores both modalities as uint8 in 0..1, quantization step 1/255. Depth
  storage error is at most ~4.9 mm, elevation storage error ~3.9 mm.

**The world's direct privileged terrain-scan input is disabled in both runs.**
This removes the geometric shortcut that made the previous depth experiment
ambiguous. Actual simulator state/action, previous action, friction and actuator
delays remain model context. Contacts are supervised predicted outputs, not force
inputs. Commands enter policy/reward/critic, not the dynamics network directly.
Analytic rewards, critic and failure head still access privileged geometry; these
are not end-to-end deployable perception-only systems. Deployment would require
real height-map estimation and state/context estimation.

Primary implementation references:

- [legged_gym local height samples and normalization](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)
- [Isaac Lab yaw-aligned grid scan](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py)

## What will establish usefulness

Every 1,000 iterations, use the same held-out environment windows and real actions
to predict 1/8/16/32 steps with intact, shuffled or mean initial perception latent.
Report physical position/velocity/joint RMSE and contact BCE, valid window counts,
image reconstruction error versus a mean-image baseline, and latent standard
deviation. Diagnostics restore CPU/CUDA RNG so they do not alter training draws.
Shuffled inputs are an intervention, not a new training distribution. Higher
error after shuffling indicates dependence/usefulness on these windows; it does
not by itself establish real-world robustness. Raw latent losses from different
encoders cannot be ranked as a common physical error metric.

After each run, evaluate three seeds x 128 first episodes x 20 s, both intact and
with shuffled perception. Report velocity/yaw tracking RMSE, survival, duration,
path length and displacement; survival is not obstacle-completion success.
Different trained policies collect different held-out state distributions, so
cross-model prediction errors are not a strictly shared-dataset benchmark.
There is one training seed per variant; evaluation seeds are not training repeats.

## Operation

Server: ai2-root-4080. Isolated worktree:
`/nix/ykj/instinct_locomotion/InstinctMJ-perception`.
Uses the existing official-uv interpreter through PYTHONPATH; no environment or
driver changes. Logs/checkpoints stay on the existing /workspace mount.

From that worktree, each command occupies one line:

```bash
bash tools/supermimic_locomotion/service_perception_20261030.sh start
bash tools/supermimic_locomotion/service_perception_20261030.sh status
```

The Supervisor worker waits for the original queue's explicit priority pause,
uses one free GPU from 0/3, and holds the shared GPU lock through training and
evaluation. Stage failures stop the queue; they do not silently skip experiments.
Old source/manifest/checkpoints are preserved. After both jobs finish, the worker
resumes the original Supervisor queue. Latest two checkpoint pairs plus the best
training-proxy pair are retained. No videos or scheduled chat notifications.

Queue state/results:
`/workspace/ykj/instinct_supermimic/logs/locomotion_official/perception_noamp_20261030`.
