import numpy as np
import pandas as pd
import os


#
# Utility functions for dataset operation
#


def get_landmark_positions(folder_name):
    """
    reads the initPoses.csv file in folder_name and returns its contents as a ndarray
    Args:
        folder_name (str): path where raw "initPoses.csv" is stored
    Returns:
        qr_poses (ndarray):
    """
    names_qr = ["id", "time", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    num_cols = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]

    lm_pos = pd.read_csv(os.path.join(folder_name, "initPoses.csv"), names=names_qr,
                         sep=";", decimal=",", header=None)

    lm_pos[num_cols] = lm_pos[num_cols].astype('float32')
    lm_pos['tz'] = lm_pos['tz'] * -1  # fixing arcore world view

    return lm_pos


def get_landmarks_pos_tensor(folder_name, recompute=True):
    """
        calculates the euclidean distance to every landmark for every frame

        Args:
            folder_name (str):
            recompute (bool): if True the calculates tensor is stored as tensor.npy in folder_name
        Returns:
            tensor: shape(n_frames, n_markers):
            landmarks: ids of markers

    """

    lm_pos = get_landmark_positions(folder_name)

    landmark_ids = lm_pos["id"].unique()
    landmark_ids.sort()

    file_path = folder_name + "/tensor.npy"

    if os.path.exists(file_path) and not recompute:
        return np.load(file_path), landmark_ids

    poses = get_trajectory(folder_name, df=True)

    tensor = np.full((len(poses), len(landmark_ids), 3), np.nan)

    lm_idx = 0
    time_lm_poses = lm_pos["time"].to_numpy()
    time_lm_ids = lm_pos["id"].to_numpy()
    lm_poses_np = lm_pos.iloc[:, 2:5].to_numpy()
    poses_np = poses.iloc[:, 1:4].to_numpy()

    camera_equals_lm_error = False

    for t_idx, t_pos in enumerate(poses["time"]):

        # positions to all previously visited landmarks are logged for each camera frame
        while lm_idx < len(time_lm_poses) and time_lm_poses[lm_idx] == t_pos:
            lm_id = int(time_lm_ids[lm_idx])
            lm_id_position = np.where(landmark_ids == lm_id)[0]
            lm_pose = lm_poses_np[lm_idx, :]

            # prevent strange error where landmark pose suddenly changes to camera pose
            if not np.array_equal(lm_pose, poses_np[t_idx, :]):
                tensor[t_idx, lm_id_position, :] = lm_pose
            else:
                camera_equals_lm_error = True

            lm_idx += 1

            if lm_idx == len(lm_pos):
                break

    if camera_equals_lm_error:
        print("Camera = landmark pose occurred during trajectory")

    np.save(file_path, tensor)

    return tensor, landmark_ids


def get_landmarks(folder_name, agg_metric='mean'):
    """
    Args:
        folder_name (str): path where raw data is stored
        agg_metric (str): 'mean' or 'median'
    Returns:
        landmarks (ndarray): array containing the x,y,z coordinates of every landmark of the trajectory,
            shape (n_labels,3)
        labels (list): of length n_labels
    """

    lm_pos = get_landmark_positions(folder_name)

    if agg_metric == 'mean':
        agg_fnc = np.mean
    elif agg_metric == 'median':
        agg_fnc = np.median
    else:
        agg_fnc = np.median
        print("unspecified aggregation function, allowed are 'mean' or 'median', using median as default")

    # find unique marker ids
    marker_ids = lm_pos['id'].unique()
    marker_ids.sort()

    landmark_dict = {id_val: agg_fnc(lm_pos[lm_pos["id"] == id_val].iloc[:, 2:5].to_numpy(), axis=0) for id_val in
                     marker_ids}

    # transform to list
    landmarks_pos = []
    landmarks_id = []
    for k, v in landmark_dict.items():
        landmarks_pos += [v]
        landmarks_id += [k]
    landmarks_pos = np.array(landmarks_pos)

    return landmarks_pos, landmarks_id


def get_trajectory(folder_name, df=False):
    """
    reads poses.csv file from given folder and returns its contents as an array

    Args:
        folder_name (str): path where raw "poses.csv" is stored
        df (bool): if true returns all files contens, if false only  returns x,y,z axis
    Returns:
        data ndarray of shape (-1,3) x,z,y
        or
        data raw pandas array
    """
    names = ["time", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    num_cols = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]

    poses = pd.read_csv(os.path.join(folder_name, "poses.csv"),
                        names=names,
                        sep=';',
                        decimal=",",
                        header=None)

    poses[num_cols] = poses[num_cols].astype('float32')
    poses['tz'] = poses['tz'] * -1  # fixing arcore world view

    res = poses.iloc[:, 1:4].to_numpy()

    if df:
        res = poses

    return res


def get_floor_from_path(path):
    try:
        floor = [int(val.split('_')[-1]) for val in path.split(os.sep) if 'floor' in val][0]
    except OSError:
        return -1

    return floor


def get_marker_dict(marker_path):
    marker_df = pd.read_csv(marker_path + "/marker_dict.csv", index_col=0)
    marker_dict = {}
    for marker_id, row in marker_df.iterrows():
        marker_dict[marker_id] = np.array([row.x, row.y])
    return marker_dict
