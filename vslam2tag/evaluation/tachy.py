import math

from vslam2tag.data_annotation import NANO_PER_MS
from vslam2tag.utils.floor_plan_plot_base import init_floorplan
from vslam2tag.evaluation.tachymeter_postprocessing import transform_tachy_segment
import os
import pandas as pd
import numpy as np


from vslam2tag.utils.definitions import get_project_root, TACHY_GT_COL, LOCAL_COL

root = get_project_root()


def compute_global_time_offset():
    path = root + "/evaluation/evaluation_data/floor_4/"
    off_LG = []
    off_OP = []

    for dev in ["LG_tachy", "OnePlus_tachy"]:
        p = os.listdir(path + dev)
        p.sort()
        for d in p:
            if d.startswith('.'):
                continue

            file = path + dev + "/" + d

            try:
                # compute_trajectory(file, correct_critical_frames=True, pos_jump_th=10.0)
                offset = get_pos_based_time_offset(file)
                print(file + ":" + str(offset))

                if "LG" in dev:
                    off_LG += [offset]
                else:
                    off_OP += [offset]

            except FileNotFoundError:
                print("skipping due to missing tachy data")
            except ValueError:
                print("skipping due to only nan entries in vslam pos")

    off_LG = np.array(off_LG).astype(int)
    off_OP = np.array(off_OP).astype(int)

    off_dict = {"LG": np.nanmedian(off_LG).astype(int),
                "OnePlus": np.nanmedian(off_OP).astype(int)}

    return off_dict


def plot_matched_data_of_folder(folder):
    pos = np.genfromtxt(folder + "/coords_local_post.csv", delimiter=',')

    fp = init_floorplan()
    fp.draw_points(pos[:, 0], pos[:, 1], s=2.5, color=LOCAL_COL, label='Local')

    t_pos = get_transformed_tachy_data(folder + "/tachy.csv")
    # t_pos_2 = get_transformed_tachy_data()
    fp.draw_points(t_pos[:, 0], t_pos[:, 1], s=2.5, color=TACHY_GT_COL, label='Ground truth')

    return fp


def _get_manually_transformed_tachy_data(file="20220214_Tracking.csv", start_idx=0):
    df = pd.read_csv(file, delimiter=";")
    pos = df.iloc[start_idx:, 2:4].to_numpy()

    pos[:, 0] += 10.7
    pos[:, 1] += 7.3

    return pos


def get_transformed_tachy_data(file="20220214_Tracking.csv", start_idx=0, manually=False):
    if manually:
        return _get_manually_transformed_tachy_data(file, start_idx=start_idx)
    else:
        return transform_tachy_segment(file)


def time_based_error(folder, offset=0, metric='mean', mapping_type='local'):
    # convert time to datetime
    time = np.genfromtxt(folder + "/coords_{}_post_time.csv".format(mapping_type), delimiter=',')
    time = np.array([np.datetime64(int(t / NANO_PER_MS), 'ms') for t in time])
    vslam_time = np.array([np.datetime64(t, 'ns') for t in time])

    if not math.isnan(offset):
        vslam_time -= offset
    vslam_pos = np.genfromtxt(folder + "/coords_{}_post.csv".format(mapping_type), delimiter=',')
    tachy_pos = get_transformed_tachy_data(folder + "/tachy.csv")
    df = pd.read_csv(folder + "/tachy.csv", delimiter=";")
    tachy_time = pd.to_datetime(df.time).to_numpy()

    last_match = -1
    dists = []
    pred_pos = []
    times = []

    for t_idx, t_time in enumerate(tachy_time):
        diff = np.abs(vslam_time - t_time)
        closest_idx = np.argmin(diff)
        time_offset = diff[closest_idx]
        dist = np.linalg.norm(tachy_pos[t_idx] - vslam_pos[closest_idx])
        # print("dist: {}, time: {}".format(dist, time_offset))
        if closest_idx == last_match:
            print(closest_idx)
            break
        last_match = closest_idx
        if not np.isnan(dist):
            dists += [dist]
            times += [t_time]
            pred_pos += [vslam_pos[closest_idx]]

    dists = np.array(dists)
    times = np.array(times)
    if metric == 'mean':
        return np.mean(dists)
    elif metric == 'median':
        return np.median(dists)
    elif metric == 'raw':
        return dists, times
    else:
        return np.mean(dists)


def get_pos_based_time_offset(folder, dist_th=0.2):

    # convert time to datetime
    time = np.genfromtxt(folder + "/coords_local_raw_post_time.csv", delimiter=',')
    time = np.array([np.datetime64(int(t / NANO_PER_MS), 'ms') for t in time])
    vslam_time = np.array([np.datetime64(t, 'ns') for t in time])

    vslam_pos = np.genfromtxt(folder + "/coords_local_raw_post.csv", delimiter=',')
    tachy_pos = get_transformed_tachy_data(folder + "/tachy.csv")

    df = pd.read_csv(folder + "/tachy.csv", delimiter=";")
    tachy_time = pd.to_datetime(df.time).to_numpy()

    time_offsets = []
    for tp_idx, tp in enumerate(tachy_pos):
        dist = np.linalg.norm(vslam_pos - tp, axis=1)
        closest_idx = np.nanargmin(dist)
        dist = dist[closest_idx]
        try:
            time_offset = vslam_time[closest_idx] - tachy_time[tp_idx]
        except IndexError:
            print("break")
        if dist < dist_th:
            time_offsets += [time_offset]
        # print("dist: {}, time: {}".format(dist, time_offset))

    mean_offset = np.median(np.array(time_offsets))

    return mean_offset
