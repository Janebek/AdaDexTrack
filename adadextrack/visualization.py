"""Matched camera recordings of semantic groups, without changing simulated states."""

import json
import math
import os
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CAMERA_POSITION = (1.08, -0.82, 0.84)
CAMERA_TARGET = (0.43, 0.07, 0.25)
CAMERA_FOV = 45.0
COLORS = {
    "stamp": (224, 112, 62),
    "toothpaste": (52, 153, 196),
    "wineglass": (218, 170, 59),
    "camera": (52, 153, 196),
    "alarmclock": (148, 117, 208),
    "apple": (224, 112, 62),
    "hammer": (218, 170, 59),
    "torusmedium": (63, 177, 139),
}


@lru_cache(maxsize=16)
def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


class GroupRecorder:
    """One camera per reference; identical local viewpoints in replay and physics."""

    def __init__(self, gym, sim, envs, entries, output, mode,
                 position=CAMERA_POSITION, target=CAMERA_TARGET, fov=CAMERA_FOV,
                 width=640, height=480, fps=12, expected_frames=300,
                 title=None, subtitle=None, panel_labels=None):
        from isaacgym import gymapi

        self.gym, self.sim, self.envs = gym, sim, envs
        self.entries, self.output = entries, Path(output)
        self.expected_frames = expected_frames
        self.frames = 0
        self.closed = False
        if len(envs) != len(entries) or not entries:
            raise ValueError("Each reference must have its own camera/environment")
        if len({str(entry.group_id) for entry in entries}) != 1:
            raise ValueError("A recording contains exactly one semantic group")
        if self.output.exists():
            raise FileExistsError(self.output)
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("Install FFmpeg to record example videos")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = width, height
        self.columns = min(2, len(entries))
        self.rows = math.ceil(len(entries) / self.columns)
        self.gap, self.header, self.label = 8, 64, 30
        self.canvas_width = self.columns * (width + self.gap) + self.gap
        self.canvas_height = self.header + self.rows * (height + self.label + self.gap) + self.gap
        self.accent = COLORS.get(entries[0].object_name, (224, 112, 62))
        self.mode = mode
        self.title = title or "Group {} | {}".format(entries[0].group_id, entries[0].object_name)
        self.subtitle = subtitle or ("Raw reference replay" if mode == "replay" else "Specialist | physics simulation")
        self.panel_labels = panel_labels if panel_labels is not None else [
            "Reference {}".format(entry.reference_id) for entry in entries]
        if len(self.panel_labels) != len(entries):
            raise ValueError("Each reference must have a panel label")
        self.api = gymapi
        self.cameras = []
        gym.set_light_parameters(sim, 0, gymapi.Vec3(.65, .65, .62),
                                 gymapi.Vec3(.35, .37, .40), gymapi.Vec3(.7, -.6, 1.5))
        gym.set_light_parameters(sim, 1, gymapi.Vec3(.20, .22, .25),
                                 gymapi.Vec3(0, 0, 0), gymapi.Vec3(-1, 1, 1))
        for light in (2, 3):
            gym.set_light_parameters(sim, light, gymapi.Vec3(0, 0, 0),
                                     gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 1))
        # Only appearance changes. No root states, contacts or dynamics are modified.
        for env in envs:
            actor = gym.find_actor_handle(env, "object")
            if actor >= 0:
                for body in range(gym.get_actor_rigid_body_count(env, actor)):
                    gym.set_rigid_body_color(env, actor, body, gymapi.MESH_VISUAL,
                                             gymapi.Vec3(*(value / 255 for value in self.accent)))
            props = gymapi.CameraProperties()
            props.width, props.height = width, height
            props.horizontal_fov = fov
            props.supersampling_horizontal = 2
            props.supersampling_vertical = 2
            camera = gym.create_camera_sensor(env, props)
            if camera < 0:
                raise RuntimeError("Could not create a recording camera")
            gym.set_camera_location(camera, env, gymapi.Vec3(*position), gymapi.Vec3(*target))
            self.cameras.append(camera)
        self.metadata = {
            "mode": mode, "group_id": str(entries[0].group_id),
            "object": entries[0].object_name,
            "reference_ids": [str(entry.reference_id) for entry in entries],
            "camera_position": list(position), "camera_target": list(target),
            "horizontal_fov": fov, "camera_size": [width, height],
            "video_size": [self.canvas_width, self.canvas_height], "fps": fps,
        }
        self.partial = self.output.with_name(self.output.stem + ".partial.mp4")
        self.log = self.output.with_suffix(".ffmpeg.log").open("w")
        encoder_env = os.environ.copy()
        # Use the encoder's own libraries, rather than Isaac Gym's Conda libs.
        encoder_env.pop("LD_LIBRARY_PATH", None)
        self.process = subprocess.Popen([
            executable, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
            "{}x{}".format(self.canvas_width, self.canvas_height),
            "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-threads", "2", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(self.partial),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log, env=encoder_env)

    def capture(self, frame):
        if frame != self.frames:
            raise ValueError("Recording must retain every frame in order")
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        image = Image.new("RGB", (self.canvas_width, self.canvas_height), (235, 240, 246))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.canvas_width, self.header), fill=(24, 38, 55))
        draw.text((16, 10), self.title, font=font(22, True), fill=(255, 255, 255))
        draw.text((16, 38), self.subtitle, font=font(15), fill=(191, 205, 221))
        for index, (env, camera, entry) in enumerate(zip(self.envs, self.cameras, self.entries)):
            pixels = self.gym.get_camera_image(self.sim, env, camera, self.api.IMAGE_COLOR)
            pixels = np.asarray(pixels, dtype=np.uint8).reshape(self.height, self.width, 4)
            x = self.gap + (index % self.columns) * (self.width + self.gap)
            y = self.header + self.gap + (index // self.columns) * (self.height + self.label + self.gap)
            draw.rectangle((x, y, x + self.width, y + self.label), fill=(255, 255, 255))
            draw.rectangle((x, y, x + 4, y + self.label), fill=self.accent)
            draw.text((x + 12, y + 5), self.panel_labels[index],
                      font=font(15, True), fill=(44, 61, 80))
            image.paste(Image.fromarray(pixels[:, :, :3]), (x, y + self.label))
        if frame in (0, 75, 150, 225, self.expected_frames - 1):
            image.save(self.output.with_name(self.output.stem + "_frame_{:03d}.png".format(frame)))
        self.process.stdin.write(np.asarray(image).tobytes())
        self.frames += 1

    def close(self, require_complete=True):
        if self.closed:
            return
        self.closed = True
        self.process.stdin.close()
        try:
            code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("FFmpeg did not finish the recording")
        finally:
            self.log.close()
        if code != 0:
            raise RuntimeError("FFmpeg failed; see " + str(self.output.with_suffix(".ffmpeg.log")))
        complete = self.frames == self.expected_frames
        self.metadata.update(frames=self.frames, status="complete" if complete else "incomplete")
        self.output.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        if not complete and require_complete:
            raise RuntimeError("Incomplete example video: {} frames".format(self.frames))
        if complete:
            self.partial.replace(self.output)

