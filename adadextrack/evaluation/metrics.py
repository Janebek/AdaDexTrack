"""Shared AdaDexTrack evaluation: L2 translation, unit quaternions, 300 frames.

This NumPy-only module is also shipped unchanged in the planner repository and
the standalone release. It does not import Isaac Gym or alter training rewards.
"""

import math
import numpy as np

FRAMES = 300
ACTION_DIM = 22
LINKS = ("palm_lower", "thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head")
JOINT_INVERSE = np.argsort(list(range(10)) + list(range(14, 22)) + [10, 11, 12, 13])
ERROR_FIELDS = (
    "object_pos_error_m", "object_rot_error_rad", "wrist_pos_error_m",
    "wrist_rot_error_rad", "finger_mae_rad",
)
POSITION_THRESHOLD_M = 0.10
ROTATION_THRESHOLD_RAD = math.radians(60)
FINGER_THRESHOLD_RAD = math.radians(40)


def frame_array(value, width, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (1, FRAMES, width):
        array = array[0]
    if array.shape != (FRAMES, width):
        raise ValueError("{}: expected ({}, {}), got {}".format(name, FRAMES, width, array.shape))
    if not np.isfinite(array).all():
        raise ValueError("{} contains non-finite values".format(name))
    return array


def unit_quaternions(value, name):
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or not np.isfinite(norm).all() or np.any(norm < 1e-8):
        raise ValueError("{} contains invalid quaternions".format(name))
    return value / norm


def load_reference(path):
    """Read the same frame-aligned metric targets for online and offline use."""
    try:
        data = np.load(str(path), allow_pickle=True).item()
        pos = frame_array(data["object_transl"], 3, "object_transl")
        rot = unit_quaternions(frame_array(data["object_rot_quat"], 4, "object_rot_quat"), "object_rot_quat")
        qpos = frame_array(data["robot_delta_states_weights_np"], ACTION_DIM, "reference qpos")
        positions = data["link_key_to_link_pos"]
        rotations = data["link_key_to_link_rot"]
        for link in LINKS:
            frame_array(positions[link], 3, link + " position")
        palm_pos = frame_array(positions["palm_lower"], 3, "palm position")
        palm_rot = unit_quaternions(frame_array(rotations["palm_lower"], 4, "palm rotation"), "palm rotation")
        return dict(object_pose=np.concatenate((pos, rot), axis=-1),
                    hand_pose=np.concatenate((palm_pos, palm_rot), axis=-1), qpos=qpos[:, JOINT_INVERSE])
    except Exception as error:
        raise ValueError("Invalid reference {}: {}".format(path, error)) from error


def quaternion_error(prediction, reference):
    prediction = unit_quaternions(prediction, "predicted rotation")
    reference = unit_quaternions(reference, "reference rotation")
    dot = np.abs(np.sum(prediction * reference, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def tracking_errors(obj_poses, hand_poses, qpos, reference):
    """Return five errors per frame or per environment, with any batch shape."""
    obj, hand, qpos = [np.asarray(value, dtype=np.float64) for value in (obj_poses, hand_poses, qpos)]
    targets = [np.asarray(reference[key], dtype=np.float64) for key in ("object_pose", "hand_pose", "qpos")]
    for actual, target, width in zip((obj, hand, qpos), targets, (7, 7, ACTION_DIM)):
        if actual.shape != target.shape or actual.ndim < 1 or actual.shape[-1] != width:
            raise ValueError("Recorded and target states have incompatible shapes")
        if not np.isfinite(actual).all() or not np.isfinite(target).all():
            raise ValueError("Non-finite recorded or target state")
    if obj.shape[:-1] != hand.shape[:-1] or obj.shape[:-1] != qpos.shape[:-1]:
        raise ValueError("Recorded state batch dimensions differ")
    target_obj, target_hand, target_qpos = targets
    return np.stack((
        np.linalg.norm(obj[..., :3] - target_obj[..., :3], axis=-1),
        quaternion_error(obj[..., 3:7], target_obj[..., 3:7]),
        np.linalg.norm(hand[..., :3] - target_hand[..., :3], axis=-1),
        quaternion_error(hand[..., 3:7], target_hand[..., 3:7]),
        np.abs(qpos[..., 6:22] - target_qpos[..., 6:22]).mean(axis=-1),
    ), axis=-1)


def metrics_from_errors(errors):
    errors = np.asarray(errors, dtype=np.float64)
    if errors.shape != (FRAMES, len(ERROR_FIELDS)) or not np.isfinite(errors).all():
        raise ValueError("Metrics require exactly 300 finite frames")
    position = 0.5 * (errors[:, 0] + errors[:, 2])
    rotation = 0.5 * (errors[:, 1] + errors[:, 3])
    mask = ((position <= POSITION_THRESHOLD_M) & (rotation <= ROTATION_THRESHOLD_RAD)
            & (errors[:, 4] <= FINGER_THRESHOLD_RAD))
    means = {name: float(errors[:, index].mean()) for index, name in enumerate(ERROR_FIELDS)}
    means.update(num_frames=FRAMES,
        mean_error_success=bool(position.mean() <= POSITION_THRESHOLD_M
            and rotation.mean() <= ROTATION_THRESHOLD_RAD and errors[:, 4].mean() <= FINGER_THRESHOLD_RAD),
        completion_rate=float(mask.mean()), completed_frames=int(mask.sum()))
    return means


def compute_paper_metrics(obj_poses, hand_poses, qpos, reference):
    obj = frame_array(obj_poses, 7, "recorded object pose")
    hand = frame_array(hand_poses, 7, "recorded palm pose")
    qpos = frame_array(qpos, ACTION_DIM, "recorded qpos")
    targets = {key: frame_array(reference[key], width, "target " + key)
               for key, width in (("object_pose", 7), ("hand_pose", 7), ("qpos", ACTION_DIM))}
    return metrics_from_errors(tracking_errors(obj, hand, qpos, targets))


def aggregate_metrics(rows, expected_ids):
    if not rows:
        raise ValueError("No evaluated references")
    ids = [str(row["reference_id"]) for row in rows]
    expected_ids = [str(value) for value in expected_ids]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
        raise ValueError("Evaluated reference coverage differs from the requested CSV")
    for row in rows:
        if row["num_frames"] != FRAMES:
            raise ValueError("Incomplete rollout: {}".format(row["reference_id"]))
        if not all(np.isfinite(row[name]) for name in ERROR_FIELDS + ("completion_rate",)):
            raise ValueError("Non-finite metric: {}".format(row["reference_id"]))
    metrics = {name: float(np.mean([row[name] for row in rows])) for name in ERROR_FIELDS}
    metrics.update(mean_error_success_rate=float(np.mean([row["mean_error_success"] for row in rows])),
                   completion_rate=float(np.mean([row["completion_rate"] for row in rows])))
    return metrics


def print_summary(metrics):
    print("\nT_o (cm) | R_o (rad) | T_h (cm) | R_h (rad) | E_finger (rad) | Mean-error (%) | Completion (%)")
    print("{:.4f} | {:.6f} | {:.4f} | {:.6f} | {:.6f} | {:.4f} | {:.4f}".format(
        100 * metrics["object_pos_error_m"], metrics["object_rot_error_rad"],
        100 * metrics["wrist_pos_error_m"], metrics["wrist_rot_error_rad"],
        metrics["finger_mae_rad"], 100 * metrics["mean_error_success_rate"],
        100 * metrics["completion_rate"],
    ), flush=True)


class PaperMetricRecorder:
    """Accumulate one synchronized evaluation episode without storing states.

    References use the same loader as offline evaluation. Each append consumes
    progress 1..300 and records only the five errors. A summary is returned at
    frame 300. Missing, duplicated or reordered frames fail explicitly.
    """

    def __init__(self, reference_ids, reference_paths):
        self.reference_ids = [str(value) for value in reference_ids]
        if not self.reference_ids or len(set(self.reference_ids)) != len(self.reference_ids):
            raise ValueError("Paper evaluation requires one environment per reference")
        if len(reference_paths) != len(self.reference_ids):
            raise ValueError("Reference paths and IDs differ in length")
        references = [load_reference(path) for path in reference_paths]
        self.targets = {key: np.stack([ref[key] for ref in references], axis=1)
                        for key in ("object_pose", "hand_pose", "qpos")}
        self.errors = np.empty((FRAMES, len(self.reference_ids), len(ERROR_FIELDS)), dtype=np.float64)
        self.frames = 0
        self.rows = None

    def append(self, progress, obj_poses, hand_poses, qpos):
        progress = np.asarray(progress)
        if progress.shape != (len(self.reference_ids),) or not np.all(progress == self.frames + 1):
            raise ValueError("Paper metrics require synchronized progress 1..300 without resets")
        if self.frames >= FRAMES:
            raise ValueError("Start a new recorder for the next evaluation episode")
        target = {key: value[self.frames] for key, value in self.targets.items()}
        self.errors[self.frames] = tracking_errors(obj_poses, hand_poses, qpos, target)
        self.frames += 1
        if self.frames < FRAMES:
            return None
        self.rows = [dict(reference_id=ref_id, **metrics_from_errors(self.errors[:, index]))
                     for index, ref_id in enumerate(self.reference_ids)]
        return aggregate_metrics(self.rows, self.reference_ids)
