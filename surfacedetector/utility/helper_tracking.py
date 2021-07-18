# import ipdb
import numpy as np
import logging
import bisect
from scipy.spatial.transform import Rotation as R

# T265 to D400
# H_t265_d400 = np.array([
#     [1, 0, 0, 0],
#     [0, -1.0, 0, 0],
#     [0, 0, -1.0, 0],
#     [0, 0, 0, 1]])

# H_Standard_t265 = np.array([
#     [1, 0, 0, 0],
#     [0, 0, -1, 0],
#     [0, 1, 0, 0],
#     [0, 0, 0, 1]])


# Global variables to hold T265 Pose
# Should hold 0.5 seconds of data at 200 HZ 
T265_ROTATION = []
T265_TIMES = []
MAX_POSES = 100


def value_within(number, goal, left=10, right=10):
    min_left = goal - left
    min_right = goal + right

    before =  number < min_left
    within =  number >= min_left and number <= min_right
    after = number > min_right

    return after


def cycle_pose_frames(t265_pipeline, ts_depth):
    """Will cycle through T265 Pose frames until timstamps are equal

    Args:
        t265_pipeline (rs.pipeline): Realsense Pipeline
        ts_depth (float): Timestamp, milliseconds since epoch

    Returns:
        float: Time stamp of current pose
    """
    within = False
    t265_frames = None
    ts_t265 = None
    while not within:
        try:
            t265_frames = t265_pipeline.wait_for_frames(timeout_ms=6)
        except Exception:
            break
        if t265_frames:
            pose_frame = t265_frames.get_pose_frame()
            ts_t265 = callback_pose(pose_frame)
            within = value_within(ts_t265, ts_depth)
        else:
            within = True
    return ts_t265

def callback_pose(frame):
    """Takes a pose frame and extract 6DOF data and stores in a list

    Args:
        frame (rs.frame): RealSense pose frame

    Returns:
        float: timestamp of frame
    """
    global T265_TRANSLATION, T265_ROTATION
    try:
        ts = frame.get_timestamp()
        domain = frame.frame_timestamp_domain
        pose = frame.as_pose_frame()
        # import ipdb; ipdb.set_trace()
        data = pose.get_pose_data()
        t = data.translation
        r = data.rotation
        T265_ROTATION.append([r.x, r.y, r.z, r.w])
        T265_TIMES.append(int(ts))
        if len(T265_TIMES) >= MAX_POSES:
            T265_ROTATION.pop(0)
            T265_TIMES.pop(0)
        return ts
    except Exception:
        logging.exception("Error in callback for pose")
    return 0.0

def get_pose_index(ts_):
    ts = int(ts_)
    idx = bisect.bisect_left(T265_TIMES, ts, lo=0)
    return min(idx, len(T265_TIMES) - 1)


def get_pose_matrix(ts_):
    """Gets the "best" rotation of the T265 at the given timestamp

    Args:
        ts_ (float): Timestamp, milliseconds since epoch

    Returns:
        [np.ndarray]: Euler angles (degrees), zxy
    """
    logging.debug("Get Pose at %r", int(ts_))
    idx = get_pose_index(ts_)
    quat = T265_ROTATION[idx]
    ts = T265_TIMES[idx]

    euler_t265 = R.from_quat(quat).as_euler("zxy", degrees=True)
    # extrinsic = H_t265_W @ H_t265_d400[:3, :3]
    logging.debug("Frame TimeStamp: %r; Pose TimeStamp %r", int(ts_), ts)
    return euler_t265
