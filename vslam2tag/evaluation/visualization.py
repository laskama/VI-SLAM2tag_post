from pathlib import Path

from matplotlib.patches import ConnectionPatch

from vslam2tag.data_annotation import NANO_PER_S
from vslam2tag.evaluation.metrics import compute_annotation_error
from vslam2tag.evaluation.tachy import plot_matched_data_of_folder
from vslam2tag.trajectory_mapping import subtrajectory_detection
from vslam2tag.utils.dataset_utilities import get_floor_from_path, get_trajectory, get_landmarks_pos_tensor, \
    get_marker_dict
from vslam2tag.utils.definitions import get_project_root, LOCAL_CORR_COL, GLOBAL_COL, LOCAL_COL, RAW_TRAJ_COL, \
    CUT_TRAJ_COL, MAP_LINE_COL, TACHY_GT_COL
from vslam2tag.utils.floor_plan_plot_base import init_floorplan
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from vslam2tag.utils.plotting_utils import align_trajectory

root = get_project_root()


def create_tachy_table(df):
    print(df[["error_local", "error_local_corrected"]].describe().round(decimals=2).to_latex())


def create_ref_point_table(df):
    df_sub = df[df["data_retained"] == 1]

    print(df[["error_local", "error_local_corr", "error_global"]].describe().round(decimals=2).to_latex())
    print(df_sub[["error_local", "error_global"]].describe().round(decimals=2).to_latex())


def visualization_of_tachy_trajectory(path="/results/evaluation_data/floor_4/OnePlus_tachy/2022-01-14T16:01:04"):

    fp = plot_matched_data_of_folder(path)
    fp.filename = "output/paper/tachy_trajectory.pdf"

    gt_line = mlines.Line2D([], [], color=TACHY_GT_COL, label='Ground truth')
    local_line = mlines.Line2D([], [], color=LOCAL_COL, label='Mapped')
    lngdh = fp.axis.legend(handles=[local_line, gt_line], bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                       mode="center", borderaxespad=0, ncol=2)#, title="Marker idx")
    # change the marker size manually for both lines
    for lh in lngdh.legendHandles:
        lh._sizes = [20]


def ref_point_plot(df, plot_type='bar'):

    errors_global = df["error_global"].to_numpy()
    errors_local = df["error_local"].to_numpy()
    errors_local_corr = df["error_local_corr"].to_numpy()
    marker_movement = df["mmm"].apply(np.mean).to_numpy()

    order = np.argsort(marker_movement)
    errors_global = errors_global[order]
    errors_local = errors_local[order]
    errors_local_corr = errors_local_corr[order]
    marker_movement = marker_movement[order]
    trajectory_ids = (np.arange(len(order))+1)[order]

    plt_dic = {"x": np.concatenate((np.arange(len(errors_local_corr)),
                                    np.arange(len(errors_local)),
                                    np.arange(len(errors_global)))),
               "y": np.concatenate((errors_local_corr, errors_local, errors_global)),
               "Mapping Type": ["Local (corrected)"] * len(errors_local) +
                               ["Local"] * len(errors_local) +
                               ["Global (LS)"] * len(errors_global)}

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.grid(False)

    if plot_type == 'line':
        ax = sns.lineplot(data=plt_dic, x="x", y="y", hue="Mapping Type", ax=ax, markers=True, marker='o', palette=[LOCAL_CORR_COL, LOCAL_COL, GLOBAL_COL])
    elif plot_type == 'bar':
        ax = sns.barplot(data=plt_dic, x="x", y="y", hue="Mapping Type", ax=ax, palette=[LOCAL_CORR_COL, LOCAL_COL, GLOBAL_COL])
    ax2.plot(np.arange(len(marker_movement)), marker_movement, linestyle='--', color=GLOBAL_COL)
    ax.set_xlabel('Trajectory ID (sorted by internal landmark movement)')
    ax.set_ylabel('Mean-error [m] (at markers)')
    ax.legend(title='Mapping Type', loc='upper left')
    ax.set_yticks(np.arange(0, 14.01, 1.0))
    ax.set_xticklabels(trajectory_ids)
    ax2.set_ylabel('Mean landmark update (ARCore) [m] (--)')
    plt.savefig("output/paper/Refpoint_error_{}.pdf".format(plot_type), bbox_inches='tight')


def show_trajectory_of_long_term_plot(path="/data/floor_1/LG_ref/2022-01-19T16:06:28"):

    floor_id = 1

    # initialize plot
    fp_local = init_floorplan(floor=floor_id, filename='output/paper/drift_local.pdf', strict_cutoff=True)

    # local visualization
    c_local = np.genfromtxt(path + "/coords_local_post.csv", delimiter=',')
    fp_local.draw_points(c_local[:, 0], c_local[:, 1], alpha=0.1, color=LOCAL_COL, s=.5,
                         label="local")
    compute_annotation_error(path, floor_plotter=fp_local, show_plot=True, floor=floor_id, mapping_type='local')

    # global visualization
    fp_global = init_floorplan(floor=floor_id, filename='output/paper/drift_global.pdf', strict_cutoff=True)
    c_global = np.genfromtxt(path + "/coords_global_post.csv", delimiter=',')
    fp_global.draw_points(c_global[:, 0], c_global[:, 1], alpha=0.1, color=GLOBAL_COL, s=.5,
                              label="global")
    compute_annotation_error(path, floor_plotter=fp_global, show_plot=True, floor=floor_id, mapping_type='global')


def visualize_subtrajectory_detection(path="OnePlus_tachy/2022-01-14T16:33:27"):
    plt.figure(figsize=[6.4, 3.0])
    sns.set_theme(style="whitegrid")
    pal = sns.color_palette("deep", n_colors=3 + 2)

    lm_pos, lm_ids = get_landmarks_pos_tensor(path, recompute=True)

    traj = get_trajectory(path)

    plot_details = subtrajectory_detection(traj, lm_pos, return_plot_info=True)
    dist = plot_details['dist']

    above_idx = range(len(dist))

    for m_id in range(len(lm_ids)):

        plt.plot(above_idx, dist[above_idx, m_id], color=pal[m_id], label=str(lm_ids[m_id]))

        sub_idx = plot_details[m_id]['sub_idx']
        sub_parts = np.split(sub_idx, np.where(np.diff(sub_idx) != 1)[0] + 1)
        for part in sub_parts:
            plt.plot(part, dist[part, m_id], color='black')

        min_vals = plot_details[m_id]['min_vals']
        for m in min_vals:
            plt.axvline(m, linestyle='--', color='black')

    lngdh = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                       mode="expand", borderaxespad=0, ncol=3, title="Marker idx")
    # change the marker size manually for both lines
    for lh in lngdh.legendHandles:
        lh._sizes = [20]

    plt.xlabel("Index of camera frame")
    plt.ylabel("Distance of camera pose to markers [m]")


def visualize_local_transformation(path, plot_subtrajectory=None):

    traj = get_trajectory(path)
    traj_df = get_trajectory(path, df=True)
    lm_pos, lm_ids = get_landmarks_pos_tensor(path)
    sub_frames, sub_lms = subtrajectory_detection(traj, lm_pos)

    idx_del = np.where(sub_lms == -1)
    sub_lms = np.delete(sub_lms, idx_del)
    sub_frames = np.delete(sub_frames, idx_del)

    marker_path = str(Path(path).parent.parent.resolve())
    marker_dict = get_marker_dict(marker_path)

    traj, marker = align_trajectory(traj_df, lm_pos, sub_frames, sub_lms)

    if plot_subtrajectory is not None:
        sub_lms = sub_lms[plot_subtrajectory]
        sub_frames = sub_frames[plot_subtrajectory]
        marker = marker[plot_subtrajectory, :]

    fig = plt.figure()

    gs = fig.add_gridspec(3, len(sub_frames)-1)

    raw_ax = fig.add_subplot(gs[0, :])
    transformed_ax = fig.add_subplot(gs[2, :])

    axes = [fig.add_subplot(gs[1, i]) for i in range(len(sub_frames)-1)]

    axes = axes[::-1]

    floor = get_floor_from_path(path)
    floor_plotter = init_floorplan(floor=floor, show_markers=True, axis=transformed_ax, annotate=True)
    c_local = np.genfromtxt(path + "/coords_local_post.csv", delimiter=',')
    floor_plotter.draw_points(c_local[:, 0], c_local[:, 1], alpha=0.1, color=LOCAL_COL, s=.5,
                              label="local")

    raw_ax.plot(traj[sub_frames[0]:sub_frames[-1], 0], traj[sub_frames[0]:sub_frames[-1], 2], color=RAW_TRAJ_COL)

    raw_ax.axis('equal')
    raw_ax.get_xaxis().set_visible(False)
    raw_ax.get_yaxis().set_visible(False)

    x = marker[:, 0]
    y = marker[:, 1]

    raw_ax.scatter(x, y, marker="x", s=50, color=RAW_TRAJ_COL)
    for i in range(len(axes)):
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)

        axes[i].axis("equal")
        axes[i].set_facecolor("none")

        axes[i].plot(traj[sub_frames[i]:sub_frames[i + 1], 0], traj[sub_frames[i]:sub_frames[i + 1], 2], color=CUT_TRAJ_COL)
        axes[i].scatter(x[i:i+2], y[i:i+2], marker="x", s=50, color=CUT_TRAJ_COL)

    # draw connections between plots
    for idx in range(len(axes)):
        print(idx)
        con = ConnectionPatch(xyA=(x[idx+1], y[idx+1]),
                              xyB=(x[idx+1], y[idx+1]), coordsA="data", coordsB="data",
                              axesA=axes[idx], axesB=raw_ax,
                              linestyle="solid",
                              color=MAP_LINE_COL)

        axes[idx].add_artist(con)

        con = ConnectionPatch(xyA=(x[idx], y[idx]),
                              xyB=(x[idx], y[idx]), coordsA="data", coordsB="data",
                              axesA=axes[idx], axesB=raw_ax,
                              linestyle="solid",
                              color=MAP_LINE_COL)

        axes[idx].add_artist(con)

        con = ConnectionPatch(xyA=(marker_dict[lm_ids[sub_lms[idx]]][0], marker_dict[lm_ids[sub_lms[idx]]][1]),
                              xyB=(x[idx], y[idx]), coordsA="data", coordsB="data",
                              axesA=transformed_ax, axesB=axes[idx],
                              arrowstyle='<-',
                              linestyle="dashed",
                              color=MAP_LINE_COL)

        axes[idx].add_artist(con)

        con = ConnectionPatch(xyA=(marker_dict[lm_ids[sub_lms[idx+1]]][0], marker_dict[lm_ids[sub_lms[idx+1]]][1]),
                              xyB=(x[idx+1], y[idx+1]), coordsA="data", coordsB="data",
                              axesA=transformed_ax, axesB=axes[idx],
                              arrowstyle='<-',
                              linestyle="dashed",
                              color=MAP_LINE_COL)

        axes[idx].add_artist(con)


def plot_landmark_configuration(floor=1):
    init_floorplan(floor=floor, show_markers=True)


def plot_annotated_of_single_trajectory(path, floor_plotter=None, show_markers=False, annotate=False, mapping_mode='local'):
    if floor_plotter is None:
        floor_plotter = init_floorplan(floor=get_floor_from_path(path), show_markers=show_markers, annotate=annotate)

    coords_post = np.genfromtxt(path + "/coords_{}_post.csv".format(mapping_mode), delimiter=',')
    floor_plotter.draw_points(coords_post[:, 0], coords_post[:, 1], color="red", s=5, label='IMU data')

    if os.path.exists(path + "/wifi_annotated.csv"):
        csi = pd.read_csv(path + "/wifi_annotated.csv")
        coords = csi[["x_coord", "y_coord"]].to_numpy()
        floor_plotter.draw_points(coords[:, 0], coords[:, 1], s=5, color="blue", label='WLAN fingerprints', alpha=0.5)

    floor_plotter.axis.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=14,
                              fancybox=True, shadow=False, ncol=5)

    return floor_plotter


def visualize_dynamic_rss_collection_of_single_scans(dev="OnePlus_2", floor=1, folder=None, time_interal=120):
    id_offset = 0
    dfs = []

    base_path = root + "/data/floor_{}/".format(floor)
    for d in os.listdir(base_path + dev):
        fp = base_path + dev + "/" + d
        print(fp)
        if not os.path.exists(fp + "/wifi_annotated.csv") or (folder is not None and folder not in fp):
            continue

        df = pd.read_csv(fp + "/wifi_annotated.csv")
        df["id"] += id_offset
        id_offset = df["id"].max(axis=0)

        if time_interal is not None:
            start_time = df['time'].min()
            end_time = start_time + time_interal * NANO_PER_S
            df = df[df["time"].between(start_time, end_time)]

        dfs += [df]

    df = pd.concat(dfs, axis=0)
    print("duration [s]: {}".format((df['time'].max() - df['time'].min()) / NANO_PER_S))

    num_fp = df["id"].unique()

    floor_plotter = init_floorplan(floor=floor, filename='output/paper/dynamic_fp_labels.pdf')
    colors = sns.color_palette(n_colors=len(num_fp))
    for c_idx, idx in enumerate(num_fp):
        sub = df[df["id"] == idx]
        pos = sub[["x_coord", "y_coord"]].to_numpy()

        floor_plotter.draw_points(pos[:, 0], pos[:, 1], color=colors[c_idx])

        # draw mean positions
        floor_plotter.draw_points(np.mean(pos[:, 0]), np.mean(pos[:, 1]), color='black', marker='*', s=50)

    # floor_plotter.set_title("Number of fingerprints: {}".format(len(num_fp)))
    floor_plotter.save_plot(bbox_inches='tight')
    floor_plotter.show_plot()


def visualize_trajectories(path, show_local=True, show_global=True, show_discarded_data=False, save_fig=False):
    fig, axes = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [.8, 1]}, dpi=200)
    plt.subplots_adjust(hspace=0.05)
    sns.set_theme(style='white')

    floor = get_floor_from_path(path)
    floor_plotter = init_floorplan(floor=floor, show_markers=True, axis=axes[1], annotate=False)
    floor_plotter.axis.grid(False)
    floor_plotter.axis.set_xticks([])
    floor_plotter.axis.set_yticks([])

    if show_local:
        c_local = np.genfromtxt(path + "/coords_local_post.csv", delimiter=',')
        floor_plotter.draw_points(c_local[:, 0], c_local[:, 1], alpha=0.1, color=LOCAL_CORR_COL, s=.5,
                                  label="local")

        false_idx = np.where(np.any(np.isnan(c_local), axis=1))[0]

    if show_global:
        c_global = np.genfromtxt(path + "/coords_global_post.csv", delimiter=',')
        floor_plotter.draw_points(c_global[:, 0], c_global[:, 1], alpha=0.1, color=GLOBAL_COL, s=.5,
                                  label="global")

    if show_local and show_global:
        blue_line = mlines.Line2D([], [], color=GLOBAL_COL, label='Global (LS)')
        reds_line = mlines.Line2D([], [], color=LOCAL_CORR_COL, label='Local (corrected)')

        axes[1].legend(handles=[blue_line, reds_line], loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       fancybox=True, shadow=False, ncol=5)

    data = get_trajectory(path)
    traj_df = get_trajectory(path, df=True)
    tensor, _ = get_landmarks_pos_tensor(path)

    sub_traj, labels = subtrajectory_detection(data, tensor)

    data, marker = align_trajectory(traj_df, tensor, sub_traj, labels)

    axes[0].plot(data[:, 0], data[:, 2], color='black')

    if show_discarded_data:
        axes[0].scatter(data[false_idx, 0], data[false_idx, 2], color='red', s=5)

    axes[0].scatter(marker[:, 0], marker[:, 1], color='black', s=10)
    axes[0].axis('equal')
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[0].set_adjustable("box")

    if save_fig:
        traj_path = root + "/evaluation/output/trajectories/"
        if not os.path.exists(traj_path):
            os.makedirs(traj_path)
        plt.savefig(os.path.join(path, "trajectory.pdf"), bbox_inches='tight')
        plt.savefig(traj_path + "{}.pdf".format(path.split('/')[-1]), bbox_inches='tight')
