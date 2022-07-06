import pandas as pd
import numpy as np
import os

from vslam2tag.trajectory_mapping import get_trajectory, subtrajectory_detection, get_landmarks_pos_tensor
from vslam2tag.utils.definitions import get_project_root

WIFI_FILE_HEADER = ["id", "time", "type", "ssid", "mac", "rss"]
SENSORS_FILE_HEADER = ["time", "type", "v1", "v2", "v3", "v4", "v5", "v6"]
NANO_PER_MS = 1000000
NANO_PER_S = NANO_PER_MS * 1000
SEC_PER_MIN = 60

root = get_project_root()


#
#   Time-based merging of collected data (WiFi & IMU) and postprocessed trajectory
#


def merge(path, imu=True, rss=True, mapping_base='local'):
    """
    Annotates the data that was collected during a trajectory
    Args:
        path: The path to the collected trajectory
        imu: Whether to annotate IMU data
        rss: Whether to annotate WiFi scans
        mapping_base: The postprocessed trajectory that is used (options: 'local' or 'global')

    Returns:
    None - annotated data are stored as separate files
    """
    # get trajectory and rss dataframes
    traj_df = get_trajectory(path, df=True)
    rss_df = pd.read_csv(path + "/wifi.csv", delimiter=";", names=WIFI_FILE_HEADER)

    # get IMU dataframe
    dec = "." if "OnePlus" in path else ","
    sensors_df = pd.read_csv(path + "/sensors.csv", delimiter=";", decimal=dec, names=SENSORS_FILE_HEADER)
    sensors_df[SENSORS_FILE_HEADER[2:]] = sensors_df[SENSORS_FILE_HEADER[2:]].astype(float)

    # get post-processed trajectory coordinates and corresponding timestamps
    coords = np.genfromtxt(path + "/coords_{}_post.csv".format(mapping_base), delimiter=',')
    c_time = np.genfromtxt(path + "/coords_{}_post_time.csv".format(mapping_base), delimiter=',')

    # to find bounds (trajectory until first landmark and trajectory after last landmark are discarded)
    sub_traj, labels = subtrajectory_detection(
        trajectory=get_trajectory(path),
        tensor=get_landmarks_pos_tensor(path, recompute=False)[0])

    # delete same sub trajectories as done in trajectory computation
    idx_del = np.where(np.diff(labels) == 0)[0]
    sub_traj = np.delete(sub_traj, idx_del + 1)
    traj_df = traj_df.iloc[sub_traj[0]:sub_traj[-1], :]

    # remove data that was collected outside recorded trajectory window
    time_bounds = traj_df.iloc[[0, -1], :]["time"].to_list()

    rss_df["time"] *= 1000
    rss_df = rss_df[rss_df["time"].between(*time_bounds)]
    sensors_df = sensors_df[sensors_df["time"].between(*time_bounds)]

    if rss:
        _merge(rss_df, traj_df, coords, c_time, filename=path + "/wifi_annotated.csv")

    if imu:
        _merge(sensors_df, traj_df, coords, c_time, filename=path + "/sensors_annotated.csv")


def _merge(data_df, traj_df, coords, c_time, filename="/wifi_annotated.csv",
           data_time_key="time", traj_time_key="time", timestamp_warning_bound_ms=100):
    """
    Annotates the data_df with the closest position (time-based) of traj_df
    Args:
        data_df: Holds the data which is about to be annotated
        traj_df: Holds the positions of the postprocessed trajectory
        coords: Coordinates of the postprocessed trajectory
        c_time: Timestamps of the postprocessed trajectory
        filename: Filename for storing annotated data
        data_time_key: key of data_df that holds time values
        traj_time_key: key of data_df that holds time values
        timestamp_warning_bound_ms: If the time is of merged entries is larger than this bound,
                                    a warning is produced

    Returns:
    None - Annotated data are stored as separate files
    """
    data_df = data_df.sort_values(by=[data_time_key])
    time_data = data_df[data_time_key].to_numpy()
    time_traj = traj_df[traj_time_key].to_numpy()

    matched_coords = []
    traj_idx = 0
    for csi_idx in range(len(data_df)):

        while traj_idx < len(time_traj) and csi_idx < len(time_data) and time_traj[traj_idx] < time_data[csi_idx]:
            traj_idx += 1

        # no more labeled positions available for merging => fill remaining with nan
        if traj_idx >= len(time_traj):
            matched_coords += [np.array([np.nan, np.nan])]
            continue

        # found matching coordinate
        matched_coords += [coords[traj_idx, :]]

        # verification if timestamps matched correctly
        matched_time_coord = c_time[traj_idx]
        matchted_time_df = time_traj[traj_idx]
        time_match_diff = (matched_time_coord - time_data[csi_idx]) / 1000000
        if matched_time_coord != matchted_time_df:
            print("matching problem")
        if time_match_diff > timestamp_warning_bound_ms:
            print("matching time diff: {}".format(time_match_diff))

    if len(matched_coords) > 0:
        matched_coords = np.concatenate(matched_coords, axis=0).reshape(-1, 2)
        data_df["x_coord"] = matched_coords[:, 0]
        data_df["y_coord"] = matched_coords[:, 1]
        data_df.to_csv(filename, index=False)


#
#   Get annotated WiFi fingerprinting dataset (average position tag per scan)
#

def get_scan_based_rss_dataset_of_phone(path="OnePlus_2", mac_addr=None):
    """
    Obtain labeled WiFi dataset by using the mean annotated positions of a single WiFi scan
    to globally annotate the entire scan (fingerprint)
    Args:
        path: The path to the collected trajectories
        mac_addr: optional numpy array that holds the mac_addr which should be used. If None => use sorted appearance of
                  mac addresses to generate RSS vector

    Returns:
    Tuple of (RSS matrix, Position matrix, trajectories, scan_ids (for each row of matrix), mac_addr (columns of matrix)
    """
    df = get_rss_df_of_phone(path)

    # exclude scans without position
    null_pos_scans = df[["x_coord", "y_coord"]].isnull().any(axis=1)
    df = df[~null_pos_scans]

    if mac_addr is None:
        mac_addr = df["mac"].unique()

    scan_ids = df["id"].unique()

    rss = np.full((len(scan_ids), len(mac_addr)), -110.0)
    pos = np.zeros((len(scan_ids), 2))
    trajectories = []

    for idx, id in enumerate(scan_ids):
        sub = df[df["id"] == id]
        positions = sub[["x_coord", "y_coord"]].to_numpy()
        trajectories += [positions]
        pos_avg = np.mean(positions, axis=0)
        pos[idx, :] = pos_avg
        for _, s in sub.iterrows():
            mac_idx = np.where(mac_addr == s["mac"])[0]
            rss[idx, mac_idx] = s["rss"]

    return rss, pos, trajectories, scan_ids, mac_addr


def get_rss_df_of_phone(path="OnePlus_2"):
    """
    Obtain annotated WiFi scans as dataframe
    Args:
        path: The path to the collected trajectories

    Returns:
    Dataframe that holds annotated WiFi scans
    """
    id_offset = 0
    dfs = []
    for d in os.listdir(path):
        fp = path + "/" + d

        if not os.path.exists(fp + "/wifi_annotated.csv"):
            continue

        df = pd.read_csv(fp + "/wifi_annotated.csv")
        df["id"] += id_offset
        id_offset = df["id"].max(axis=0)
        dfs += [df]
    df = pd.concat(dfs, axis=0)

    df = df.sort_values(by=["time", "id"])

    return df
