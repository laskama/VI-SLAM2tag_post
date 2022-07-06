from pathlib import Path
import os
import seaborn as sns

# color definitions
col_val = sns.color_palette("deep", n_colors=5)
LOCAL_CORR_COL = col_val[0]
LOCAL_COL = col_val[1]
GLOBAL_COL = col_val[2]
LANDMARK_COL = 'black'

MAP_LINE_COL = 'grey'
RAW_TRAJ_COL = 'black'
CUT_TRAJ_COL = 'black'

TACHY_GT_COL = 'black'


def get_project_root():
    return str(Path(__file__).parent.parent)


def get_cache_path():
    path = get_project_root() + "/cached/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path
