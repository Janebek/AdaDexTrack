"""Convert a saved ROS1 easy_handeye eye-on-base result to deployment settings."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from examples.sim2real.interfaces import rigid_transform


def load_eye_to_hand(path):
    calibration = yaml.safe_load(Path(path).expanduser().read_text())
    parameters = calibration["parameters"]
    if parameters.get("eye_on_hand") is not False:
        raise ValueError("Expected an eye-to-hand calibration with eye_on_hand: false")
    base, camera = parameters["robot_base_frame"], parameters["tracking_base_frame"]
    if not isinstance(base, str) or not base or not isinstance(camera, str) or not camera:
        raise ValueError("Calibration must identify the robot base and camera frames")
    values = calibration["transformation"]
    translation = np.array([values[key] for key in ("x", "y", "z")], dtype=np.float64)
    quaternion = np.array([values[key] for key in ("qx", "qy", "qz", "qw")], dtype=np.float64)
    if not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-8:
        raise ValueError("Invalid calibration quaternion")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform[:3, 3] = translation
    return dict(base_frame=base, camera_frame=camera,
                T_base_camera=rigid_transform(transform).tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path, help="Saved easy_handeye eye-on-base YAML")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="Deployment YAML to write")
    args = parser.parse_args(argv)
    if args.output.expanduser().resolve() == args.calibration.expanduser().resolve():
        parser.error("Output must differ from the saved calibration")
    settings = yaml.safe_load(args.config.expanduser().read_text())
    settings.update(load_eye_to_hand(args.calibration))
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(settings, sort_keys=False))
    print("Saved {}: {} <- {}".format(output, settings["base_frame"], settings["camera_frame"]))


if __name__ == "__main__":
    main()
