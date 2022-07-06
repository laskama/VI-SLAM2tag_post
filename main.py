from absl import flags, app
import os

from vslam2tag.evaluation.visualization import visualize_trajectories
from vslam2tag.trajectory_mapping import map_trajectory
from vslam2tag.data_annotation import merge
from vslam2tag.utils.definitions import get_project_root

root = get_project_root()

flags.DEFINE_string('data_path', root + "/data/", 'Path where training dataset is found.')
flags.DEFINE_string('mapping', 'local', 'Results of which mapping strategy should be used for dataset annotation (either local (recommended) or global).')
flags.DEFINE_string('global_transformation_type', 'similarity', 'Coordinate transformation used for global mapping strategy, either "similarity" or "affine"')
flags.DEFINE_boolean('annotate', True, 'Whether to annotate sensor data with position via time-based matching')
flags.DEFINE_boolean('visualize', True, 'Whether to store a visualization of the mapped trajectories')
flags.DEFINE_list('floors', [0, 1, 2, 3, 4], 'List of floors that should be considered.')
flags.DEFINE_list('devices', ["OnePlus", "LG", "S20", "Galaxy"], 'List of devices that should be considered.')
flags.DEFINE_float('pos_jump_th', 5.0, 'Position jump threshold for discarding data of locally mapped trajectory')

FLAGS = flags.FLAGS


def main(argv):

    for floor in FLAGS.floors:
        for device in FLAGS.devices:
            path = os.path.join(FLAGS.data_path, "floor_" + str(floor), device)

            for traj in os.listdir(path):
                if traj.startswith("."):  # catches .DS_store
                    continue

                traj_path = str(os.path.join(path, traj))
                print(traj_path)

                # compute various mapping with different strategies
                map_trajectory(folder_name=traj_path,
                               pos_jump_th=FLAGS.pos_jump_th,
                               global_transformation_type=FLAGS.global_transformation_type)

                # annotate data using the mapped trajectories from the chosen mapping type
                if FLAGS.annotate:
                    merge(traj_path, imu=True, rss=True, mapping_base=FLAGS.mapping)

                # store visualization of the mapped trajectories
                if FLAGS.visualize:
                    visualize_trajectories(traj_path, show_discarded_data=True, save_fig=True)


if __name__ == '__main__':
    app.run(main)
