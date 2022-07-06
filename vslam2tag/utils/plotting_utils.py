from shapely.geometry import LineString

from vslam2tag.trajectory_mapping import calc_mapping


def align_trajectory(traj_df, tensor, sub_traj, landmark_ids):
    """
    Aligns trajectory roughly horizontal (for clean visualization)
    Args:
        traj_df: The trajectory dataframe
        tensor: The landmark position tensor
        sub_traj: The segment of the specified subtrajectory
        landmark_ids: The landmark ids

    Returns:
    """

    start_idx = sub_traj[0]
    end_idx = sub_traj[-1]
    poses = traj_df.iloc[:, 1:4].to_numpy()

    x0, y0 = poses[start_idx, 0], poses[start_idx, 2]
    x1, y1 = poses[end_idx, 0], poses[end_idx, 2]

    marker_points = LineString(zip(tensor[sub_traj, landmark_ids, 0], tensor[sub_traj, landmark_ids, 2]))

    if x0 > x1:
        x0_true, y0_true = 100, 100
        x1_true, y1_true = 110, 100
    else:
        x0_true, y0_true = 110, 100
        x1_true, y1_true = 100, 100

    transformed_line, transformed_points = calc_mapping(traj_df, x0, y0, x1, y1, x0_true, y0_true, x1_true, y1_true,
                                                        marker_points=marker_points, pad_zaxis=True)

    return transformed_line, transformed_points
