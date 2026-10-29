"""Export all scalar samples and readable training curves from a completed run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from instinct_mj.utils.smooth_rewards import smooth_dont_wait_penalty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ea = EventAccumulator(str(args.run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    series = {tag: ea.Scalars(tag) for tag in ea.Tags()["scalars"]}
    with (args.output_dir / "scalars.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "iteration", "wall_time", "value"])
        for tag, samples in series.items():
            writer.writerows((tag, x.step, x.wall_time, x.value) for x in samples)
    milestones = [500, 1000, 2000, 4000, 6000, 8000]
    selected = ["sim/reward", "sim/episode_length", "sim/terminated_fraction", "sim/terrain_level",
                "wm/err_8step", "latent/err_8step", "style/d_policy", "style/d_reference",
                "imagination/reward_term/style", "imagination/reward_term/p_fail_mean"]
    summary = {
        "run": str(args.run_dir), "note": "Original hard-dont_wait run; training statistics, not evaluation success.",
        "milestones": {str(it): {tag: next((x.value for x in series[tag] if x.step == it), None)
                                 for tag in selected} for it in milestones},
        "last_1000_iterations_mean": {
            tag: float(np.mean([x.value for x in series[tag] if 7000 < x.step <= 8000])) for tag in selected
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    panels = [
        ("Simulator task reward (excludes AMP)", [("sim/reward", "task reward", 1)], "reward / s", False),
        ("Episode duration (maximum 20 s)", [("sim/episode_length", "duration", .02)], "seconds", False),
        ("Failures among ended training episodes", [("sim/terminated_fraction", "failure fraction", 100)], "%", False),
        ("Terrain curriculum keeps getting harder", [("sim/terrain_level", "mean terrain level", 1)], "level (0-9)", False),
        ("World-model 8-step state error", [("wm/err_8step", "state error", 1)], "normalized error", True),
        ("Visual-latent 8-step error", [("latent/err_8step", "latent error", 1)], "normalized error", True),
        ("AMP discriminator on REAL trajectories", [("style/d_policy", "policy", 1), ("style/d_reference", "reference", 1)], "score (targets -1 / +1)", False),
        ("AMP style reward on IMAGINED trajectories", [("imagination/reward_term/style", "weighted style reward", 1)], "reward / step", False),
        ("Learned failure probability in imagination", [("imagination/reward_term/p_fail_mean", "p_fail", 100)], "% per step", False),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    for ax, (title, curves, ylabel, log) in zip(axes.flat, panels):
        for tag, label, scale in curves:
            samples = series[tag]
            x = np.array([p.step for p in samples])
            y = np.array([p.value * scale for p in samples])
            smooth = np.array([np.mean(y[max(0, i - 4):i + 1]) for i in range(len(y))])
            line, = ax.plot(x, smooth, label=label, linewidth=1.8)
            ax.plot(x, y, color=line.get_color(), alpha=.22, linewidth=.8)
        ax.axvspan(0, 500, color="#64748b", alpha=.1)
        ax.set(title=title, xlabel="training iteration", ylabel=ylabel, xlim=(0, 8000))
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("G1 perceptive SuperMimic locomotion + AMP | 2026-10-28, seed 42\n"
                 "Original hard reward; faint = logged samples, solid = trailing 5 samples; shaded = WM-only warmup",
                 fontsize=14)
    fig.savefig(args.output_dir / "training_curves.png", dpi=160)
    fig.savefig(args.output_dir / "training_curves.svg")
    plt.close(fig)

    velocity = torch.linspace(-.5, .6, 601, dtype=torch.float64, requires_grad=True)
    command = torch.full_like(velocity, .6)
    reward = -.5 * smooth_dont_wait_penalty(velocity, command)
    grad, = torch.autograd.grad(reward.sum(), velocity)
    hard = -.5 * sum((velocity.detach() < threshold).double() for threshold in (.15, 0, -.15))
    v = velocity.detach().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(v, hard.numpy(), label="original hard reward", color="#94a3b8", linestyle="--")
    axes[0].plot(v, reward.detach().numpy(), label="smooth reward, forward command 0.6 m/s")
    axes[0].axhline(0, color="#22c55e", linestyle=":", label="standing / reached target: reward = 0")
    axes[0].set(xlabel="body-frame forward speed (m/s)", ylabel="weighted reward (before dt)", title="Same intent: discourage stopping and reversing")
    axes[1].plot(v, grad.numpy(), color="#d97706", label="smooth reward gradient")
    axes[1].axhline(0, color="#94a3b8", linestyle="--", label="hard reward gradient, away from jumps")
    axes[1].set(xlabel="body-frame forward speed (m/s)", ylabel="d reward / d speed", title="Positive gradient encourages forward acceleration")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(args.output_dir / "dont_wait_reward.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
