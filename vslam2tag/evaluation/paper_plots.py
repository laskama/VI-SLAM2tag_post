import matplotlib.pyplot as plt

from vslam2tag.evaluation.metrics import get_ref_point_dataframe, get_tachy_dataframe
from vslam2tag.evaluation.visualization import visualize_trajectories, plot_annotated_of_single_trajectory, \
    plot_landmark_configuration, visualize_local_transformation, visualize_subtrajectory_detection, \
    show_trajectory_of_long_term_plot, ref_point_plot, visualization_of_tachy_trajectory, create_ref_point_table, \
    create_tachy_table, visualize_dynamic_rss_collection_of_single_scans
from vslam2tag.trajectory_mapping import map_trajectory
from vslam2tag.utils.definitions import get_project_root

root = get_project_root()

#
#   Reproduce all paper figures and tables
#
#   - Requires that giaIndoorLoc_raw is placed in "data/" folder and was postprocessed according to README
#   - Requires that evaluation_data is placed in "evaluation/evaluation_data/" and was postprocessed according to README
#


def fig_1():
    path = root + '/data/floor_1/S20/2021-12-20T13:25:23'
    plot_landmark_configuration(floor=1)
    visualize_trajectories(path=path, show_local=True, show_global=False, show_discarded_data=True)
    plot_annotated_of_single_trajectory(path=path)


def fig_2():
    path = root + "/evaluation/evaluation_data/floor_4/Subtrajectory_split/2022-01-14T16:35:30"
    visualize_subtrajectory_detection(path=path)


def fig_3():
    path = root + "/evaluation/evaluation_data/floor_4/Subtrajectory_split/2022-01-14T16:35:30"
    map_trajectory(path)
    visualize_local_transformation(path)


def fig_4():
    path = root + "/evaluation/evaluation_data/floor_1/OnePlus_ref/2022-01-19T15:51:00"
    map_trajectory(path)
    visualize_trajectories(path)


def fig_5():
    path = root + "/evaluation/evaluation_data/floor_1/LG_ref/2022-01-19T16:06:28"
    map_trajectory(path)
    show_trajectory_of_long_term_plot(path=path)


def fig_6():
    df = get_ref_point_dataframe(floors=[1, 4], devices=['LG_ref', 'OnePlus_ref'], cache_file="dataframe",
                                 to_excel=True)
    ref_point_plot(df)


def fig_7():
    path = root + "/evaluation/evaluation_data/floor_4/OnePlus_tachy/2022-01-14T16:01:04"
    visualization_of_tachy_trajectory(path)


def fig_9():
    visualize_dynamic_rss_collection_of_single_scans(dev="Galaxy", floor=4, folder="2021-12-20T14:17:21")


def table_1():
    df = get_ref_point_dataframe(floors=[1, 4], devices=['LG_ref', 'OnePlus_ref'], cache_file="dataframe_similarity",
                                 to_excel=True)
    create_ref_point_table(df)


def table_2():
    t_df = get_tachy_dataframe(floors=[4], devices=["LG_tachy", "OnePlus_tachy"],
                               experiment_name="tachymeter_evaluation")

    create_tachy_table(t_df)


if __name__ == '__main__':
    fig_1()
    fig_2()
    fig_3()
    fig_4()
    fig_5()
    fig_6()
    fig_7()
    fig_9()

    table_1()
    table_2()

    plt.show()