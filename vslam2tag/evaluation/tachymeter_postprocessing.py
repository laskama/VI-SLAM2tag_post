from vslam2tag.trajectory_mapping import calc_mapping
from vslam2tag.utils.dataset_utilities import get_marker_dict
from pathlib import Path
import pandas as pd


def align_cs(trajectory, points, true_points):
    """ """

    x0, y0 = points.X[0], points.Y[0]
    x1, y1 = points.X[1], points.Y[1]

    x0_true, y0_true = true_points.X[0], true_points.Y[0]
    x1_true, y1_true = true_points.X[1], true_points.Y[1]

    trajectory = trajectory.rename(columns={'X': 'tx', 'Y': 'tz'})
    transformed_line, transformed_points = calc_mapping(trajectory, x0, y0, x1, y1, x0_true, y0_true, x1_true, y1_true)

    return transformed_line, transformed_points


def get_ref_points():
    folder_name = "evaluation_data/floor_4/tachymeter"
    data = pd.read_csv(folder_name + "/20220214_Tracking.csv", sep=";")

    ref_points = data.iloc[0:3, :]
    ref_points = ref_points.iloc[1:, :].reset_index()

    return ref_points


def get_true_points():
    folder_name = "evaluation_data/floor_4/tachymeter"
    marker_path = str(Path(folder_name).parent.resolve())
    marker_dict = get_marker_dict(marker_path)
    # transformation based on markers 9 and 10
    true_points = pd.DataFrame.from_dict(marker_dict, orient='index', columns=('X', 'Y'))
    true_points = true_points.loc[[10, 9], :].reset_index()

    return true_points


def transform_tachy_segment(path):
    data = pd.read_csv(path, sep=";")
    aligned_data, _ = align_cs(data, get_ref_points(), get_true_points())
    return aligned_data
