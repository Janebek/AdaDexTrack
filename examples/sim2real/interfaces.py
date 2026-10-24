"""Small hardware/pose interfaces, with connections made only by explicit calls.

The xArm position-to-velocity controller is adapted from BunnyVisionPro (MIT).
LEAP joint mapping follows the AdaDexTrack LEAP wrapper (MIT). See LICENSES.txt.
"""

import threading
import time

import numpy as np


HAND_COMMAND_ORDER = np.array([1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7])
HAND_STATE_ORDER = np.argsort(HAND_COMMAND_ORDER)
HAND_ZERO = 3.14159  # LEAP ROS cmd_allegro conversion uses this same offset.
ARM_HZ = 250


def vector(value, width):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (width,) or not np.isfinite(value).all():
        raise ValueError("Expected {} finite joint values".format(width))
    return value.copy()


def hand_command(qpos):
    """Policy radians -> cmd_allegro radians; the driver adds its own zero."""
    return vector(qpos, 16)[HAND_COMMAND_ORDER]


def hand_state(native_position):
    """Native motor radians -> policy radians, including the joint permutation."""
    return (vector(native_position, 16) - HAND_ZERO)[HAND_STATE_ORDER]


class ArmPD:
    def __init__(self):
        self.previous_error = None

    def velocity(self, error):
        error = vector(error, 6)
        previous = error if self.previous_error is None else self.previous_error
        result = 3.0 * error + .15 * (error - previous) / (1.0 / ARM_HZ)
        self.previous_error = error
        overshoot = np.max(np.abs(result) / 2.0)
        return result / overshoot if overshoot > 1.0 + 1e-4 else result


class XArmVelocityController:
    """Supply six position targets; an explicit start launches the 250 Hz loop."""

    def __init__(self, arm, target_timeout=.5):
        self.arm, self.target_timeout = arm, target_timeout
        self.pid = ArmPD()
        self.error = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._target = None
        self._target_time = None

    @staticmethod
    def _check(code, operation):
        if code != 0:
            raise RuntimeError("xArm {} failed with code {}".format(operation, code))

    def read_position(self):
        if self.error is not None:
            raise RuntimeError("xArm control thread stopped") from self.error
        code, state = self.arm.get_joint_states(is_radian=True)
        self._check(code, "get_joint_states")
        positions = np.asarray(state[0])
        if positions.shape not in ((6,), (7,)):
            raise ValueError("Expected six active xArm joints")
        return vector(positions[:6], 6)

    def set_target(self, target):
        target = vector(target, 6)
        if self.error is not None:
            raise RuntimeError("xArm control thread stopped") from self.error
        with self._lock:
            self._target, self._target_time = target, time.monotonic()

    def _send_velocity(self, velocity):
        # The SDK's joint-velocity API takes seven slots; xArm6 uses the first six.
        self._check(self.arm.vc_set_joint_velocity(
            np.append(vector(velocity, 6), 0.).tolist(), is_radian=True,
        ), "vc_set_joint_velocity")

    def _tick(self):
        with self._lock:
            target = None if self._target is None else self._target.copy()
            timestamp = self._target_time
        if target is None or time.monotonic() - timestamp > self.target_timeout:
            raise TimeoutError("xArm position target is missing or stale")
        code, state = self.arm.get_state()
        self._check(code, "get_state")
        if state in (4, 5):
            raise RuntimeError("xArm is stopped")
        current = self.read_position()
        velocity = self.pid.velocity(target - current)
        if not self._stop.is_set():
            self._send_velocity(velocity)

    def _run(self):
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                self._tick()
                deadline += 1.0 / ARM_HZ
                delay = deadline - time.monotonic()
                if delay < 0:
                    deadline = time.monotonic()
                else:
                    self._stop.wait(delay)
        except Exception as error:
            self.error = error
            self._stop.set()
        finally:
            try:
                self._send_velocity(np.zeros(6))
            except Exception as error:
                if self.error is None:
                    self.error = error

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Use a new controller after stop")
        if self._target is None:
            raise RuntimeError("Set an initial measured-position target before start")
        self._check(self.arm.motion_enable(enable=True), "motion_enable")
        self._check(self.arm.set_mode(4), "set_mode")
        self._check(self.arm.set_state(state=0), "set_state")
        self._thread = threading.Thread(target=self._run, name="xarm-velocity", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.)
            if self._thread.is_alive():
                raise RuntimeError("xArm SDK call did not finish during stop")
        self._send_velocity(np.zeros(6))


class LeapHandROS:
    def connect(self):
        import rospy
        from sensor_msgs.msg import JointState
        from leap_hand.srv import leap_position

        self.message_type = JointState
        rospy.wait_for_service("/leap_position", timeout=5.)
        self.position = rospy.ServiceProxy("/leap_position", leap_position)
        self.publisher = rospy.Publisher("/leaphand_node/cmd_allegro", JointState, queue_size=1)

    def read_position(self):
        return hand_state(self.position().position)

    def set_target(self, target):
        message = self.message_type()
        message.position = hand_command(target).tolist()
        self.publisher.publish(message)

    def close(self):
        self.publisher.unregister()
        self.position.close()


def rigid_transform(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 calibrated transform")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError("Expected a rigid transform with a proper rotation")
    return matrix


def object_in_base(T_base_camera, T_camera_object):
    """Express the FoundationPose object pose in the robot base frame, in metres."""
    return rigid_transform(T_base_camera) @ rigid_transform(T_camera_object)


def publish_object_tf(broadcaster, transform, stamp, base_frame="base_link", object_frame="object_frame"):
    """Call from the external RGB-D pose process, using the image timestamp."""
    from geometry_msgs.msg import TransformStamped
    from scipy.spatial.transform import Rotation

    transform = rigid_transform(transform)
    message = TransformStamped()
    message.header.stamp, message.header.frame_id = stamp, base_frame
    message.child_frame_id = object_frame
    xyz = transform[:3, 3]
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    message.transform.translation.x, message.transform.translation.y, message.transform.translation.z = xyz
    (message.transform.rotation.x, message.transform.rotation.y,
     message.transform.rotation.z, message.transform.rotation.w) = xyzw
    broadcaster.sendTransform(message)


class ObjectPoseTF:
    def __init__(self, base_frame="base_link", object_frame="object_frame", max_age=.25):
        self.base_frame, self.object_frame, self.max_age = base_frame, object_frame, max_age

    def connect(self):
        import rospy
        import tf2_ros

        self.ros = rospy
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

    def read_pose(self):
        message = self.buffer.lookup_transform(self.base_frame, self.object_frame,
                                              self.ros.Time(0), self.ros.Duration(.005))
        age = (self.ros.Time.now() - message.header.stamp).to_sec()
        if not np.isfinite(age) or age < -.05 or age > self.max_age:
            raise TimeoutError("Object pose timestamp is stale or outside the local clock")
        p, q = message.transform.translation, message.transform.rotation
        pose = vector([p.x, p.y, p.z, q.x, q.y, q.z, q.w], 7)
        norm = np.linalg.norm(pose[3:])
        if norm < 1e-8:
            raise ValueError("Invalid object quaternion")
        pose[3:] /= norm
        return pose
