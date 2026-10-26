# SuperMimic perceptive motion tracking

This is a new training run, not a continuation of the earlier locomotion/AMP
experiment. It uses `Instinct-Perceptive-Shadowing-G1-v0`: ten motion-terrain
pairs and six scanned terrain meshes from the user's knee-climb dataset.
The original native rewards, tracking terminations, reference resets and
failure-based motion-bin curriculum are retained. No AMP or LAFAN data is used.
The native contact penalty excludes feet and wrists, but still mildly penalizes
other contacts (weight -0.1); tracking objectives can outweigh this penalty.
There is no blanket torso-contact termination or upright locomotion reward.

## Learning procedure

The world model receives physical robot state, previous action, full 18x32 depth,
contact-history magnitudes and joint torques. It also receives paired terrain
identity and randomized friction, mass, center of mass, actuator gains and
joint calibration. Robot position is expressed in the paired terrain's frame.
Terrain identity is privileged training information, never an actor input.
The dynamics model predicts normalized increments on eight-step replay windows.
Reset transitions are excluded and windows stop at episode boundaries.

The actor receives ten proprioceptive frames and ten future motion-reference
frames, plus a CNN encoding the 18x32 depth image. It outputs joint PD actions.
Unlike the earlier parkour experiment, future references are exogenous recorded
trajectories, not model predictions. Eighteen 10 Hz reference frames are stored
in replay and interpolated over 32 imagined 50 Hz steps. Link orientations use
hemisphere-correct normalized quaternion interpolation. Sensor noise is added
to proprioception in both real collection and imagination.

All six native tracking reward terms use differentiable rigid-body FK, including
link-origin linear/angular velocities. Joint-limit and action-rate penalties
are analytic. Torque and contact penalties use their corresponding predicted
physical channels, rather than an unconstrained scalar reward head. The contact
threshold and tracking termination thresholds are smoothed for BPTT. The critic
uses imagined GAE(lambda) targets. The simulator is never differentiated.

Startup audits compare FK positions, rotations, velocities and all non-action
reward terms to native MuJoCo/Instinct calculations. The actual action-rate
penalty is checked during collection. Nonfinite loss/gradient/state stops the
run. The run stores audit.json, protocol.json, args.json, train.csv and checkpoints
containing all networks and optimizers. Protocol records the exact git revision.

## Deliberate differences and limitations

- This is a SuperMimic-style extension, not an exact reproduction of the original
  paper or a controlled comparison with native PPO.
- Fixed training-terrain identity conditions dynamics; geometry is not represented
  by a differentiable mesh renderer. Depth is predicted, not re-rendered during
  imagination. Generalization to new meshes is not established.
- Actor uses 10 proprio frames (native perceptive PPO uses 8), a motion encoder
  and depth CNN, not the native PPO policy architecture. Goal noise is omitted.
- World model width 1024, actor widths 1024/512/256, 512 simulation environments,
  256 replay steps, 256 training batch, four WM updates per iteration. These
  differ from the flat-ground reference reproduction's larger configuration.
- Smooth imagined survival uses interpolated reference states. Native terminations
  use aiming reference frames. Dataset validity gates imagination; invalid-reset,
  border and episode-time truncations are not modeled as physical failures.
- No teacher policy or existing checkpoint bootstraps the run. Reference-state
  resets and native adaptive motion-bin sampling provide contact-rich coverage.

## Run

Each command below is one line. Environments/caches stay in /nix; data and logs
stay in /workspace. CUDA drivers and the system Python are not modified.

```bash
bash /nix/ykj/instinct_supermimic/InstinctMJ/tools/supermimic/run_tracking.sh 2026-10-26_supermimic_perceptive_tracking_seed0 --compile
```

The launcher obtains a project-specific lock and evaluates the final checkpoint.
It should be launched under nohup or a persistent terminal. The default wall-time
budget is 12 hours after environment initialization. Checkpoints are periodic;
resume explicitly with --resume <checkpoint> and use a new run name if desired.
Replay is recollected on resume; this is not a bitwise continuation.

## Evaluation

Evaluate the first full-motion episode in each environment, with motion-bin
resets disabled and reset pose perturbations disabled. Physics randomization
remains active with a fresh seed. Capture metrics inside reward computation,
before automatic reset, so falls do not contaminate contact/pose metrics with
the next episode's initial state. Report per-motion completions, failures,
censoring, root/link error and body contact counts. A timeout is not success.
The final evaluation is on the six paired training meshes, not held-out terrain.

## Deployment validation — 2026-10-26

- Mesh preprocessing required Shapely 2.2.0 and Rtree 1.4.1; both were installed
  only in the existing project uv environment. The frozen requirements now
  contain 177 packages. No drivers or system Python were changed.
- 16-environment training, two complete actor/critic/model updates, checkpoint
  reload and 16-step evaluation passed (`2026-10-26_tracking_validation_v3`).
- Production-size validation passed with 512 environments, batch 256, four WM
  batches, eight-step model training, 32-step BPTT and compiled reward/FK.
  Three complete updates and 128-environment full-motion evaluation exited 0
  (`2026-10-26_tracking_fullsize_validation_v2`). All ten motion-terrain pairs
  were represented. The nearly untrained policy failed all 128 episodes; this
  is an initialization baseline, not a trained performance claim.
- Post-collection FK maximum position error was 4.47e-6 m, velocity error
  1.79e-5 m/s, and maximum native reward discrepancy 2.98e-8. Native cached
  relative frames are refreshed for comparisons at the same post-forward pose.
- Code passes Ruff and launcher passes bash syntax validation. Validation used
  the same launcher, environment count, batch size and compile flag as production.
  No temporary test scripts were created or retained.

Production is a fresh seed-0 run named
`2026-10-26_supermimic_perceptive_tracking_seed0`, on physical GPU 0 of
`ai2-root-4080`. It has 300 world-model-only warmup iterations, then joint
training; checkpoints every 500 iterations and at the warmup boundary/exit.
Logs are CSV/JSON, with no W&B run. The launcher is detached with nohup and its
PID is recorded in the run directory. Inspect train.csv and process state to
determine current progress; this document does not imply continued liveness.
