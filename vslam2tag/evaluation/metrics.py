from pathlib import Path

import os
import numpy as np
import pandas as pd

from vslam2tag.evaluation.tachy import compute_global_time_offset, time_based_error
from vslam2tag.trajectory_mapping import subtrajectory_detection
from vslam2tag.utils.dataset_utilities import get_trajectory, get_landmarks_pos_tensor, get_landmark_positions, \
    get_landmarks
from vslam2tag.utils.definitions import get_project_root
from vslam2tag.utils.floor_plan_plot_base import init_floorplan

root = get_project_root()


def get_tachy_dataframe(floors, devices, experiment_name):
    if os.path.exists("output/paper/{}.pickle".format(experiment_name)):
        return pd.read_pickle("output/paper/{}.pickle".format(experiment_name))

    offset_dict = compute_global_time_offset()

    results = []
    for floor in floors:
        path = os.path.join(os.getcwd(), 'evaluation_data/floor_{}/'.format(floor))
        for dev in devices:
            for root, dirs, _ in os.walk(os.path.abspath(os.path.join(path, dev))):
                for dir in dirs:
                    file = os.path.join(root, dir)

                    print(file)

                    metric = 'mean'

                    # filter out those trajectories with jumping positions (e.g. when running)
                    offset = offset_dict["LG"] if "LG" in dev else offset_dict["OnePlus"]
                    error_local_filtered = time_based_error(file, offset=offset, metric=metric, mapping_type='local')

                    # raw trajectories without position jump filtering (e.g. when running)
                    offset = offset_dict["LG"] if "LG" in dev else offset_dict["OnePlus"]
                    error_local_raw = time_based_error(file, offset=offset, metric=metric, mapping_type='local_raw')

                    results.append(
                        {
                            'dev': dev,
                            'file': file,
                            'floor': floor,
                            'error_local': error_local_raw,
                            'error_local_corrected': error_local_filtered,
                        }
                    )

    results_df = pd.DataFrame(results)

    results_df.to_pickle("output/paper/{}.pickle".format(experiment_name))
    results_df.to_csv("output/paper/{}.csv".format(experiment_name))
    results_df.to_excel("output/paper/{}.xlsx".format(experiment_name))

    return results_df


def compute_marker_jumps(folder_name):
    qr_poses = get_landmark_positions(folder_name)

    landmarks, labels = get_landmarks(folder_name)

    # calculate distance to center
    means = []

    for i, label in enumerate(labels):
        center = np.array([landmarks[i, 0], landmarks[i, 2]])
        label_points = qr_poses[qr_poses["id"] == label].iloc[:, [2, 4]].to_numpy()

        label_points = label_points - center
        dist = np.linalg.norm(label_points, axis=1)

        means += [np.mean(dist)]

        # ax.annotate("%.2f\n%.2f" %(np.mean(dist), np.var(dist)), center)

    means = np.asarray(means)

    return means


def evaluate_ref_point_distance(floors, devices, experiment_name):
    results = []
    for floor in floors:
        path = os.path.join(os.getcwd(), 'evaluation_data/floor_{}/'.format(floor))
        for dev in devices:
            for root, dirs, _ in os.walk(os.path.abspath(os.path.join(path, dev))):
                for dir in dirs:
                    file = os.path.join(root, dir)

                    print(file)

                    metric = 'mean'
                    coord_l_r = np.genfromtxt(file + "/coords_local_raw_post.csv", delimiter=',')
                    num_data_l_r = len(np.where(~np.any(np.isnan(coord_l_r), axis=1))[0])
                    error_local = compute_annotation_error(file, show_plot=False, metric=metric, floor=floor, mapping_type='local_raw')

                    coord_l = np.genfromtxt(file + "/coords_local_post.csv", delimiter=',')
                    num_data_l = len(np.where(~np.any(np.isnan(coord_l), axis=1))[0])
                    error_local_corr = compute_annotation_error(file, show_plot=False,   metric=metric, floor=floor, mapping_type='local')

                    coord_g = np.genfromtxt(file + "/coords_global_post.csv", delimiter=',')
                    num_data_g = len(np.where(~np.any(np.isnan(coord_g), axis=1))[0])
                    error_global = compute_annotation_error(file, show_plot=False, metric=metric, floor=floor, mapping_type='global')

                    time_diffs = compute_annotation_error(file, show_plot=False,  metric='time', floor=floor)

                    results.append(
                        {
                            'dev': dev,
                            'file': file,
                            'floor': floor,
                            'error_local': error_local,
                            'error_local_corr': error_local_corr,
                            'error_global': error_global,
                            'data_retained': num_data_l/num_data_l_r,
                            'time_diffs': time_diffs,
                        }
                    )

    results_df = pd.DataFrame(results)

    # results_df.to_csv("output/{}.csv".format(experiment_name))
    # results_df.to_excel("output/{}.xlsx".format(experiment_name))

    return results_df


def summary(floors, devices):
    summary = []

    if devices == ["all"]:
        devices = ['LG', 'LG_tachy', 'OnePlus_2_backup', 'OnePlus_tachy', 'Galaxy',  'LG_ref',   'OnePlus', 'OnePlus_ref', 'S20']

    print(devices)
    for floor in floors:
        path = os.path.join(os.getcwd(), 'evaluation_data/floor_{}/'.format(floor))
        for dev in devices:
            for root, dirs, _ in os.walk(os.path.abspath(os.path.join(path, dev))):
                for dir in dirs:
                    folder_name = os.path.join(root, dir)

                    print(folder_name)
                    # print(dir)
                    data = get_trajectory(folder_name)
                    tensor, _ = get_landmarks_pos_tensor(folder_name)

                    sub_traj, labels = subtrajectory_detection(data, tensor)

                    idx_del = np.where(labels == -1)
                    labels_cleaned = np.delete(labels, idx_del)
                    # sub_traj = np.delete(sub_traj, idx_del)
                    n_scanned_markers = len(labels_cleaned)
                    n_unique_markers = len(np.unique(labels_cleaned))

                    idx = np.where(labels == -1)[0]
                    parts_label = np.split(labels, idx[::2]+2)
                    parts = np.split(sub_traj, idx[::2]+2)

                    npoints_local = 0
                    for i in range(len(parts_label)):
                        p = parts[i]
                        l = parts_label[i]
                        p = np.delete(p, np.where(l == -1))
                        if len(p) > 0:
                            npoints_local += p[-1]-p[0]

                    npoints_global = sub_traj[-1]-sub_traj[2] # first two labels are always [-1 -1] (before any marker is scanned)

                    poses = get_trajectory(folder_name, df=True)
                    time = poses.iloc[:, 0].to_numpy()
                    duration = (time[-1]-time[0]) / 6e+10  # ns to minutes

                    n_nans = 0.5 * np.count_nonzero(labels == -1)

                    coords = np.genfromtxt(folder_name + "/coords_local_post.csv", delimiter=',')

                    num_data = len(np.where(~np.any(np.isnan(coords), axis=1))[0])
                    coords_t = np.genfromtxt(folder_name + "/coords_local_post_time.csv", delimiter=',')

                    coords_t = coords_t / 1e+9  # ns to seconds

                    dt = np.diff(coords_t)
                    dt = dt.reshape((-1, 1))

                    ds = np.abs(np.diff(coords, axis=0))
                    v = ds / dt

                    v_total = np.hypot(v[:, 0], v[:, 1])
                    ds_total = np.hypot(ds[:, 0], ds[:, 1])

                    distance_walked = np.nansum(ds_total)
                    v_average = np.nanmean(v_total)

                    summary.append(
                        {
                            'device': dev,
                            'file': folder_name,
                            'floor': floor,
                            'duration': duration,
                            'n_nans': n_nans,
                            'n_scanned_markers': n_scanned_markers,
                            'n_unique_markers': n_unique_markers,
                            'npoints_local': npoints_local,
                            'npoints_global': npoints_global,
                            'distance_walked': distance_walked,
                            'v_average': v_average

                        }
                    )
    summary_df = pd.DataFrame(summary)

    if not os.path.exists("output/"):
        os.makedirs("output/")

    summary_df.to_excel("output/file_summary.xlsx")

    return summary_df


def evaluate_marker(floors, devices):
    results = []
    for floor in floors:
        path = os.path.join(os.getcwd(), 'evaluation_data/floor_{}/'.format(floor))
        for dev in devices:
            for root, dirs, _ in os.walk(os.path.abspath(os.path.join(path, dev))):
                for dir in dirs:
                    file = os.path.join(root, dir)
                    mmm = compute_marker_jumps(file)

                    results.append(
                        {
                            # 'dev': dev,
                            'file': file,
                            'mmm': mmm
                        }
                    )

    results_df = pd.DataFrame(results)

    # results_df.to_csv("output/{}.csv".format(experiment_name))
    # results_df.to_excel("output/{}.xlsx".format(experiment_name))

    return results_df


def get_ref_point_dataframe(floors=None, devices=None, cache_file="dataframe", to_excel=True):

    if floors is None:
        floors = [1, 4]
    if devices is None:
        devices = ["LG_ref", "OnePlus_ref"]

    if not os.path.exists("output/paper/"):
        os.makedirs("output/paper/")

    if cache_file is None or not os.path.exists("output/paper/{}.pickle".format(cache_file)):
        results_df = evaluate_ref_point_distance(floors=floors, devices=devices, experiment_name="refpoint_validation")
        summary_df = summary(floors=floors, devices=devices)

        m = evaluate_marker(floors=floors, devices=devices)
        a = pd.merge(summary_df, results_df, on='file', how='outer')
        a = pd.merge(a, m, on='file', how='outer')

        a.to_pickle("output/paper/{}.pickle".format(cache_file))

    else:
        a = pd.read_pickle("output/paper/{}.pickle".format(cache_file))

    if to_excel:
        a.to_excel("output/paper/{}.xlsx".format(cache_file))

    return a


def compute_annotation_error(path, floor_plotter=None, show_plot=True, metric='mean', floor=4, mapping_type='local'):
    dist_list = []
    time_list = []

    if show_plot and floor_plotter is None:
        floor_plotter = init_floorplan(floor=4)

    ref_marker_path = str(Path(path).parent.parent.resolve())
    marker = np.genfromtxt(ref_marker_path + "/refmarker.csv", delimiter=',')

    coords = np.genfromtxt(path + "/coords_{}_post.csv".format(mapping_type), delimiter=',')
    coords_t = np.genfromtxt(path + "/coords_{}_post_time.csv".format(mapping_type), delimiter=',')

    ref_time = pd.read_csv(path + "/refMarker.csv", delimiter=";", names=["id", "time"])
    ref_time = ref_time[ref_time["time"].between(coords_t[0], coords_t[-1])]

    unknown_pos_counter = 0
    # iterate over marker presses
    for t in ref_time["time"]:
        # find closest pos in trajectory
        idx = np.argmin(np.abs(coords_t - t))
        # get dist to all ref_points
        pos = coords[idx]
        dist = np.linalg.norm(marker - pos, axis=1)
        min_dist = np.min(dist)
        if show_plot:
            floor_plotter.draw_points(pos[0], pos[1], color="black", s=25)

        if np.isnan(min_dist):
            unknown_pos_counter += 1
        else:
            dist_list += [min_dist]
            time_list += [t]

    dist_list = np.array(dist_list)

    if metric == 'mean':
        mean_dist = np.mean(dist_list)
    elif metric == 'median':
        mean_dist = np.median(dist_list)
    elif metric == 'raw':
        mean_dist = dist_list
    elif metric == 'time':
        mean_dist = time_list
    else:
        mean_dist = np.mean(dist_list)

    if unknown_pos_counter > 0:
        print("unknown position at reference point: {}".format(unknown_pos_counter))

    if show_plot:
        floor_plotter.draw_points(marker[:, 0], marker[:, 1], color="blue", s=25, marker="*")

    return mean_dist
