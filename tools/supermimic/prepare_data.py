"""Create a named-joint AMP library from existing G1 retargeted assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--lafan", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  # LAFAN1_Retargeting_Dataset G1 CSV order (not the mjlab compiled order).
  names = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
  ]
  selected, receipt = [], []
  for source in sorted(args.source.rglob("*retargeted.npz")):
    relative = Path("contact") / source.relative_to(args.source)
    destination = args.output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
      shutil.copyfile(source, destination)
    selected.append(str(relative))
    receipt.append({"file": str(relative), "source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
  for source in sorted(args.lafan.glob("*.csv")):
    # Include locomotion as well as jumps/fall recovery; original AMP data is not publicly bundled.
    relative = Path("lafan") / (source.stem + "-retargeted.npz")
    destination = args.output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = np.loadtxt(source, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 36 or not np.isfinite(data).all():
      raise ValueError(f"Invalid G1 CSV: {source}: {data.shape}")
    if not destination.exists():
      np.savez_compressed(destination, framerate=30.0, joint_names=np.array(names),
                          joint_pos=data[:, 7:], base_pos_w=data[:, :3], base_quat_w=data[:, [6, 3, 4, 5]])
    selected.append(str(relative))
    receipt.append({"file": str(relative), "source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
  if not selected:
    raise ValueError("Empty motion library")
  (args.output / "selection.yaml").write_text(yaml.safe_dump({"selected_files": selected}, sort_keys=False))
  (args.output / "manifest.json").write_text(json.dumps({"files": receipt, "count": len(selected),
    "note": "Initial mixed AMP library; differs from original unavailable parkour_motion_without_run library."}, indent=2))
  print(f"Prepared {len(selected)} motions in {args.output}")


if __name__ == "__main__":
  main()
