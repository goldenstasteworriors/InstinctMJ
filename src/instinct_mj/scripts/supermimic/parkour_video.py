"""Stream four fixed evaluation views; freeze each first episode at termination."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


class ParkourVideo:
    def __init__(self, env, path: Path, indices: list[int], label: str, levels, kinds):
        if len(indices) != 4 or len(set(indices)) != 4 or any(i < 0 or i >= env.num_envs for i in indices):
            raise ValueError("Video requires four distinct valid environment indices")
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.env, self.path, self.indices, self.label = env, path, indices, label
        self.levels, self.kinds = levels.cpu().tolist(), kinds.cpu().tolist()
        self.frames, self.ended = {}, {}
        self.renderers = {}
        self.stride = 2
        self.font = ImageFont.load_default(size=18)
        for i in indices:
            cfg = ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY,
                               entity_name="robot", body_name="torso_link", env_idx=i,
                               max_extra_envs=0, width=640, height=360,
                               distance=3.5, elevation=-20, azimuth=135,
                               enable_shadows=False, enable_reflections=False)
            renderer = OffscreenRenderer(env.sim.mj_model, cfg, env.scene,
                                         env.sim.model, env.sim.expanded_fields)
            renderer.initialize()
            self.renderers[i] = renderer
        self.writer = imageio.get_writer(str(path), fps=1 / (env.step_dt * self.stride),
                                         codec="libx264", quality=7, pixelformat="yuv420p",
                                         macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"])

    def capture(self, step, active, done, results, reasons):
        write_frame = step % self.stride == 0
        for i in self.indices:
            if i in self.ended:
                continue
            ending = bool(active[i] and done[i])
            if write_frame or ending or i not in self.frames:
                self.renderers[i].update(self.env.sim.data)
                self.frames[i] = self.renderers[i].render().copy()
            if ending:
                self.ended[i] = (step * self.env.step_dt, results[i], list(reasons[i]))
        if not write_frame:
            return
        canvas = Image.new("RGB", (1280, 720))
        for j, i in enumerate(self.indices):
            tile = Image.fromarray(self.frames[i])
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, 640, 53), fill=(15, 23, 42))
            draw.text((8, 5), f"{self.label} | env {i} | terrain {self.kinds[i]} / level {self.levels[i]}",
                      fill="white", font=self.font)
            if i in self.ended:
                t, result, reasons_i = self.ended[i]
                caption = f"{result.upper()} at {t:.2f}s (frozen): {', '.join(reasons_i)}"
                color = "#ff8080" if result == "failure" else "#86efac"
            else:
                caption, color = f"t={step * self.env.step_dt:.2f}s | first episode", "#cbd5e1"
            draw.text((8, 29), caption, fill=color, font=self.font)
            canvas.paste(tile, ((j % 2) * 640, (j // 2) * 360))
        frame = np.asarray(canvas)
        self.writer.append_data(frame)
        if step == self.stride:
            canvas.save(self.path.with_suffix(".preview.png"))

    def close(self):
        self.writer.close()
        for renderer in self.renderers.values():
            renderer.close()
