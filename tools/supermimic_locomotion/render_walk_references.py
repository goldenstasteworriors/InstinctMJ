"""Inventory selected walking clips and render their unmodified kinematic poses.

This is reference-data playback, not a trained policy or a physics rollout.
Only a constant horizontal translation is removed for scene placement.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from instinct_mj.assets.unitree_g1 import G1_MJCF_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    selected = yaml.safe_load((args.dataset / "selection.yaml").read_text())["selected_files"]
    walks = [p for p in selected if Path(p).name.startswith("walk")]
    names = ["walk1_subject1-retargeted.npz", "walk2_subject3-retargeted.npz", "walk3_subject2-retargeted.npz"]
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for path in walks:
        with np.load(args.dataset / path) as motion:
            n, fps = len(motion["joint_pos"]), float(motion["framerate"])
        inventory.append(dict(file=path, frames=n, fps=fps, seconds=n / fps))

    xml_path = Path(G1_MJCF_PATH)
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((xml_path.parent / compiler.get("meshdir")).resolve()))
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="walk_grid", type="2d", builtin="checker",
                  rgb1="0.35 0.4 0.45", rgb2="0.25 0.3 0.35", width="512", height="512")
    ET.SubElement(asset, "material", name="walk_floor", texture="walk_grid", texrepeat="100 100", reflectance="0")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="reference_floor", type="plane", size="50 50 0.1", material="walk_floor", group="0")
    ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1", directional="true", diffuse="0.9 0.9 0.9")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    model.stat.extent = 2.0
    data = mujoco.MjData(model)
    free = int(model.jnt_qposadr[model.joint("floating_base_joint").id])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 3.0, 135, -15
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    font = ImageFont.load_default(size=22)
    videos = []
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        for name in names:
            path = next(p for p in walks if Path(p).name == name)
            motion = np.load(args.dataset / path)
            fps = float(motion["framerate"])
            n = min(len(motion["joint_pos"]), round(args.seconds * fps))
            addresses = [int(model.jnt_qposadr[model.joint(j).id]) for j in motion["joint_names"].tolist()]
            assert len(set(addresses)) == 29
            output = args.output / (Path(name).stem + ".mp4")
            if output.exists():
                raise FileExistsError(output)
            shift = motion["base_pos_w"][0].copy()
            shift[2] = 0
            with imageio.get_writer(output, fps=fps, codec="libx264", pixelformat="yuv420p",
                                    macro_block_size=1, ffmpeg_params=["-crf", "28", "-movflags", "+faststart"]) as writer:
                for i in range(n):
                    data.qpos[free:free + 3] = motion["base_pos_w"][i] - shift
                    data.qpos[free + 3:free + 7] = motion["base_quat_w"][i]
                    data.qpos[addresses] = motion["joint_pos"][i]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = data.qpos[free:free + 3]
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 960, 64), fill=(15, 23, 42))
                    draw.text((12, 7), name, fill="white", font=font)
                    draw.text((12, 35), f"REFERENCE DATA | no policy / no physics | t={i / fps:.2f}s", fill="#86efac", font=font)
                    writer.append_data(np.asarray(frame))
                    if i == n // 2:
                        frame.save(output.with_suffix(".preview.png"))
            videos.append(dict(file=str(output), source=path, start_frame=0, frames=n, fps=fps, seconds=n / fps))
            print(json.dumps(videos[-1]), flush=True)
    summary = dict(walk_files=inventory, total_frames=sum(x["frames"] for x in inventory),
                   total_seconds=sum(x["seconds"] for x in inventory), videos=videos,
                   note="Direct G1 torso-base NPZ playback; raw 30 Hz poses, no policy, physics, smoothing or contact correction.")
    (args.output / "walking_inventory.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
