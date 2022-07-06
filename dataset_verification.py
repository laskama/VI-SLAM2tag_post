import os
import pandas as pd
from absl import flags, app

from vslam2tag.utils.definitions import get_project_root

root = get_project_root()

flags.DEFINE_string('verification_path', "/data/", 'Path where verification dataset is found (giaIndoorLoc).')
FLAGS = flags.FLAGS

DATASET = root + "/data/"
WIFI_FP = "/wifi_annotated.csv"
IMU_FP = "/sensors_annotated.csv"


def verify_postprocessing(argv):

    TEST_DATASET = FLAGS.verification_path

    print("Checking whether post-processed dataset ({}) is equal to published verification dataset ({})...".format(
        DATASET, TEST_DATASET))

    for floor in [0, 1, 2, 3, 4]:
        for device in ["Galaxy", "LG", "OnePlus", "S20"]:

            p = "floor_{}/{}/".format(floor, device)

            for traj in os.listdir(TEST_DATASET + p):
                if traj.startswith('.'):
                    continue
                c_path = DATASET + p + traj
                v_path = TEST_DATASET + p + traj

                # check if files exist
                assert os.path.exists(c_path + IMU_FP), "imu annotation missing: {}".format(c_path)

                assert os.path.exists(c_path + WIFI_FP), "wifi annotation missing: {}".format(c_path)

                # check if content is equal
                imu_v = pd.read_csv(v_path + IMU_FP)
                imu_c = pd.read_csv(c_path + IMU_FP)

                wifi_v = pd.read_csv(v_path + WIFI_FP)
                wifi_c = pd.read_csv(c_path + WIFI_FP)

                assert wifi_c.equals(wifi_v), "wifi annotation differs: {}".format(c_path)

                assert imu_c.equals(imu_v), "imu annotation differs: {}".format(c_path)

    print("Check finished successfully")


if __name__ == '__main__':
    app.run(verify_postprocessing)
