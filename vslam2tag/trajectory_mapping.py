import numpy as np

from vslam2tag.utils.dataset_utilities import get_landmarks, get_landmarks_pos_tensor, get_trajectory
from vslam2tag.utils.dataset_utilities import get_marker_dict

from pathlib import Path
from shapely import affinity
from shapely.geometry import LineString
from vslam2tag.utils.helper_funcs import angle_between
from vslam2tag.utils.umeyama_transform import umeyama


def calc_mapping(segment, x0, y0, x1, y1, x0_true, y0_true, x1_true, y1_true, pad_zaxis=False, marker_points=None):
    """ calculates a transformation defined by mapping the points:
    
        (x0, y0) --> (x0_true, y0_true)
        (x1, y1) --> (x1_true, y1_true)

        the mapping is applied to the given segment and points

        Args:
            segment (pandas DataFrame): segment to transform
            x0, y0, x1, y1 (float): coordinates of base points
            x0_true, 0_true, x1_true, y1_true (float): coordinates of target points
            pad_zaxis (bool): adds a dummy z axis with every element set to 1. Defaults to False
            marker_points: optional ShapelyLinestring that holds fix points which should be mapped
        Returns:
            segment_mapped (ndarray): mapped segment of shape (-1,2) ( or (-1,3) if pad_zaxis=True)
            points_mapped (ndarray): mapped points, shape (n_points,2)
    """

    line = LineString(zip(segment.tx, segment.tz))

    if marker_points is None:
        # Create shapely points object
        points = LineString(zip([x0, x1], [y0, y1]))
    else:
        points = marker_points

    # Translation
    translated_line = affinity.translate(geom=line, xoff=x0_true - x0, yoff=y0_true - y0)
    translated_points = affinity.translate(geom=points, xoff=x0_true - x0, yoff=y0_true - y0)

    # Scaling
    length = np.linalg.norm(np.array((x0, y0)) - np.array((x1, y1)))
    length_true = np.linalg.norm(np.array((x0_true, y0_true)) - np.array((x1_true, y1_true)))

    scale_fact = length_true / length

    scaled_line = affinity.scale(geom=translated_line, xfact=scale_fact, yfact=scale_fact, origin=(x0_true, y0_true))
    scaled_points = affinity.scale(geom=translated_points, xfact=scale_fact, yfact=scale_fact,
                                   origin=(x0_true, y0_true))

    # Rotation
    angle = angle_between(vec_a=[scaled_points.xy[0][-1] - x0_true, scaled_points.xy[1][-1] - y0_true],
                          vec_b=[x1_true - x0_true, y1_true - y0_true])

    rotated_line = affinity.rotate(geom=scaled_line, angle=angle, origin=(x0_true, y0_true), use_radians=True)
    rotated_points = affinity.rotate(geom=scaled_points, angle=angle, origin=(x0_true, y0_true), use_radians=True)

    # convert Shapely objects to numpy arrays
    points_x = np.array(rotated_points.xy[0]).reshape(-1, 1)
    points_y = np.array(rotated_points.xy[1]).reshape(-1, 1)
    points_mapped = np.concatenate((points_x, points_y), axis=1)

    line_x = np.array(rotated_line.xy[0]).reshape(-1, 1)
    line_y = np.array(rotated_line.xy[1]).reshape(-1, 1)

    if pad_zaxis:
        line_z = np.ones(line_x.shape)
        segment_mapped = np.concatenate((line_x, line_z, line_y), axis=1)
    else:
        segment_mapped = np.concatenate((line_x, line_y), axis=1)

    return segment_mapped, points_mapped


def map_segment(data, tensor, start_idx, end_idx, x0_true, y0_true, x1_true, y1_true, labels=None,
                      pos_jump_th=5):
    """
        transforms the segment between two landmarks by mapping the landmarks to the corresponding reference markers
        l0 -> r0
        l1 -> r1

    Args:
        data (pandas DataFrame): raw trajectory data
        tensor (tensor): distance tensor
        start_idx (int): index in data where the segment starts
        end_idx (int): index in data where the segment ends
        x0_true (float): x coordinate of r0
        y0_true (float): y coordinate of r0
        x1_true (float): x coordinate of r1
        y1_true (float): y coordinate of r1
        labels (list, length 2):  [label of r0, label of r1]
        pos_jump_th (float): maximum allowed jump between two consecutive frames. if threshold is met the whole subtrajectory is
                            replaced with nan.

    Returns:
        segment_mapped (ndarray): coordinates of the mapped segment
        timestamps (ndarray): time entries corresponding to the coordinates
    """

    x0, y0 = tensor[start_idx, labels[0], 0], tensor[start_idx, labels[0], 2]
    x1, y1 = tensor[end_idx, labels[1], 0], tensor[end_idx, labels[1], 2]

    timestamps = data.iloc[start_idx:end_idx].time

    # skip transforming segments that contain critical frames
    if labels is not None and -1 in labels:
        return np.full((len(timestamps), 2), np.nan), timestamps

    segment = data[start_idx:end_idx]

    segment_mapped, _ = calc_mapping(segment=segment,
                                     x0=x0, y0=y0, x1=x1, y1=y1,
                                     x0_true=x0_true, y0_true=y0_true,
                                     x1_true=x1_true, y1_true=y1_true)

    # postprocessing check to identify to large gaps
    dist = np.linalg.norm(np.diff(segment_mapped, axis=0), axis=1)
    crit_jump_ids = np.where(dist > pos_jump_th)[0]

    # if a large gap is detected the whole subtrajectory is marked omitted
    if len(crit_jump_ids) > 0:
        return np.full((len(timestamps), 2), np.nan), timestamps

    return segment_mapped, timestamps


#
#   Global transformation
#


def global_transformation(traj_df, sub_traj, cache=False, cache_path=None, affine_transformation=True):
    """
        computes global transformation function based on median landmark positions and applies transformation to entire
        trajectory

    Args:
        traj_df: (pandas DataFrame): DataFrane of the raw trajectory
        sub_traj (list): index of visited landmarks.
        cache (bool): if true, results are store
        cache_path (str): path where results are saved
    Returns:
        traj_mapped (ndarray): mapped trajectory
        timestamps (ndarray): corresponding time entries
    """

    traj = traj_df.iloc[:, 1:4].to_numpy()
    timestamps = traj_df.time.to_numpy()

    # cut trajectory based on computed subtrajtories to match length local mapping strategy
    # data before the first landmark vist and after the last landmark visit are ignored
    traj = traj[sub_traj[0]:sub_traj[-1]]
    timestamps = timestamps[sub_traj[0]:sub_traj[-1]]

    # obtain the affine transformation function via least squares approach
    t_fnc, _ = get_mapping_fnc(cache_path, affine_transformation)

    # if the landmark configuration is chosen poorly (too few landmark scanned that possibly lie on one line)
    # the computed transformation matrix might not have full rank which gives bad results => try to avoid
    if t_fnc is None:
        # mapping could not be computed (matrix has not full rank)
        return None

    # transform the trajectory with the computed mapping function
    traj_mapped = t_fnc(traj[:, [0, 2]])

    if cache:
        # store post processed trajectory
        np.savetxt(cache_path + "/coords_global_post.csv", traj_mapped, delimiter=",")
        np.savetxt(cache_path + "/coords_global_post_time.csv", timestamps, delimiter=",")

    return traj_mapped, timestamps


def get_similarity_transformation_via_umeyama(primary, secondary):
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]

    # remove second column, since it contains the height
    primary = primary[:, [0, 2]]
    secondary = secondary[:, [0, 2]]

    A = umeyama(primary, secondary, estimate_scale=True)
    A = np.transpose(A)

    transform = lambda x: unpad(np.dot(pad(x), A))

    max_error = np.abs(secondary - transform(primary)).max()

    return transform, max_error


def get_affine_transformation_via_least_squares(primary, secondary):
    """
        finds optimal affine transform matrix for mapping primary matrix to the secondary matrix
        solution from:
        https://stackoverflow.com/questions/20546182/how-to-perform-coordinates-affine-transformation-using-python-part-2

    Args:
        primary (ndarray): set of virtual landmark positions
        secondary (ndarray): set of real world landmark positions

    Returns:
        transformation: transformation function
        max_error: largest error that occurs between one pair of landmarks
    """

    # remove second column, since it contains the height
    primary = primary[:, [0, 2]]
    secondary = secondary[:, [0, 2]]

    # Pad the data with ones, so that our transformation can do translations too
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]
    X = pad(primary)
    Y = pad(secondary)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    if rank < 3:
        print("Least-squares transformation has Rank < 3")
        return None, None

    if np.sign(A[0, 0]) != np.sign(A[1, 1]):
        print("reflection")

    transform = lambda x: unpad(np.dot(pad(x), A))

    max_error = np.abs(secondary - transform(primary)).max()

    return transform, max_error


def get_mapping_fnc(folder_name, affine=True):
    """
        creates the mapping function between the virtual landmark positions and the true marker positions using least squares

    Args:
      folder_name (str): path where raw data is stored
    Returns:
        t_fnc: function that maps landmarks -> markers
        max_error: largest error that occurs between one pair of landmarks
    """

    landmark_pos, landmark_id = get_landmarks(folder_name, agg_metric='median')

    marker_path = str(Path(folder_name).parent.parent.resolve())
    marker_dict = get_marker_dict(marker_path)

    true_pos = np.concatenate([marker_dict[idx][np.newaxis, :] for idx in landmark_id], axis=0)
    true_pos = np.concatenate((true_pos, np.ones((len(true_pos), 1))), axis=1)

    temp = true_pos.copy()
    true_pos[:, 1] = temp[:, 2]
    true_pos[:, 2] = temp[:, 1]

    if affine:
        t_fnc, max_error = get_affine_transformation_via_least_squares(landmark_pos, true_pos)
    else:
        t_fnc, max_error = get_similarity_transformation_via_umeyama(landmark_pos, true_pos)

    return t_fnc, max_error


def local_transformation(landmark_tensor, traj_df, sub_idx, sub_label, landmark_ids, marker_dict,
                         pos_jump_th=5, cache=False, cache_path=None, raw=False):

    """
        computes and applies local transformation for every subtrajectory by mapping the landmark positions to the
        reference markers

    Args:
        landmark_tensor (tensor): landmark distance tensor
        traj_df  (pandas DataFrame): DataFrane of the raw trajectory
        sub_idx (list): indices of visited landmarks
        sub_label (list): label of visited landmarks
        landmark_ids (list): #todo
        marker_dict (dict): dict of reference marker position in the real world
        pos_jump_th (float): maximum allowed jump between two consecutive frames. if threshold is met the whole
                            subtrajectory isreplaced with nan.
        cache (bool): if true, results are store
        cache_path (str): path where results are saved

    Returns:
        traj_mapped (ndarray): mapped trajectory
        timestamps (ndarray): corresponding time entries
    """

    # remove subparts where same marker appear twice in a row (no mapping possible)
    # [-1 -1 0 1 1 3] would be reduced to [-1 0 1 3]
    idx_del = np.where(np.diff(sub_label) == 0)[0]
    sub_label = np.delete(sub_label, idx_del + 1)
    sub_idx = np.delete(sub_idx, idx_del + 1)

    # data structure for storing mapped sub-trajectories and corresponding timestamps
    traj_list = []
    timestamps_list = []

    # detect splitting points for each landmark individually and sort lateron
    for idx in range(len(sub_idx) - 1):

        # get ground truth landmark positions
        l0 = marker_dict[landmark_ids[sub_label[idx]]]
        l1 = marker_dict[landmark_ids[sub_label[idx + 1]]]

        # compute local transformation
        seg_mapped, timestamps = map_segment(
            data=traj_df,
            tensor=landmark_tensor,
            start_idx=sub_idx[idx],
            end_idx=sub_idx[idx + 1],
            x0_true=l0[0], y0_true=l0[1],
            x1_true=l1[0], y1_true=l1[1],
            labels=[sub_label[idx], sub_label[idx + 1]],
            pos_jump_th=pos_jump_th)

        traj_list += [seg_mapped]
        timestamps_list += [timestamps]

    # store post-processed trajectory
    traj_mapped = np.concatenate(traj_list, axis=0)
    timestamps = np.concatenate(timestamps_list, axis=0)

    if cache:
        if raw:
            traj_f_name = "/coords_local_raw_post.csv"
            time_f_name = "/coords_local_raw_post_time.csv"
        else:
            traj_f_name = "/coords_local_post.csv"
            time_f_name = "/coords_local_post_time.csv"

        np.savetxt(cache_path + traj_f_name, traj_mapped, delimiter=",")
        np.savetxt(cache_path + time_f_name, timestamps, delimiter=",")

    return traj_mapped, timestamps


def subtrajectory_detection(trajectory, tensor, mark_critical_frames=True, cut_threshold=1.0, return_plot_info=False):
    """
        detects where the trajectory visits a reference marker and stores the index and id of the visit
        #todo reference to paper figure?
    Args:
        trajectory (ndarray): trajectory data
        tensor (tensor): distance tensor
        mark_critical_frames (bool): if true critical frames are detected and marked
        cut_threshold (float): distance to a landmark that must be undercut to count as a visit
    Returns:
        visit_index (list): indices in trajectory where markers were visited
        visit_label (list): label sequence of visited markers. e.g. [1,3,2,1]
    """

    # get distances to landmark for each camera frame
    dist = np.linalg.norm(trajectory[:, np.newaxis, :] - tensor, axis=2)

    num_landmarks = np.shape(tensor)[1]

    # set up data structures for storing the position and the marker ids of the computed subtrajectories
    visit_index = []
    visit_label = []

    # set up dict that holds info for generating plots
    plot_details = {idx: {} for idx in range(num_landmarks)}
    plot_details['dist'] = dist

    # set up dict that holds info for generating plots
    plot_details = {idx: {} for idx in range(num_landmarks)}
    plot_details['dist'] = dist

    for idx in range(num_landmarks):

        # get all frame ids where distance in smaller than threshold (possible intervals for splitting trajectory)
        sub_idx = np.where(dist[:, idx] < cut_threshold)[0]
        plot_details[idx]['sub_idx'] = sub_idx

        # find entry and escape idx of sub frames (identified by jump in ids)
        last = sub_idx[0] - 1
        step = [sub_idx[0]]
        min_vals = []
        for i in sub_idx:
            if i - last > 1:
                step.append(last)
                step.append(i)
            last = i

        step.append(sub_idx[-1])

        # find minimum distances in identified intervals
        try:
            for j in range(int(len(step) / 2)):
                min = np.argmin(dist[step[j * 2]:step[(j * 2) + 1], idx])
                min_vals.append(step[j * 2] + min)
        except ValueError:
            print("ValueError")

        visit_index += min_vals
        visit_label += [idx] * len(min_vals)

        plot_details[idx]['min_vals'] = min_vals

        plot_details[idx]['min_vals'] = min_vals

    # convert to numpy arrays
    visit_index = np.array(visit_index)
    visit_label = np.array(visit_label)

    # detect critical frames (no info on landmarks available, all distances are nan)
    # mark those as -1 for subsequent computations
    if mark_critical_frames:
        critical_frames = np.where(np.all(np.isnan(dist), axis=1))[0]

        # identify intervals in the critical frame via jumps of the ids
        crit_bounds = np.where(np.diff(critical_frames) > 1)[0]

        if len(critical_frames) > 0:

            # mark initial frames as critical
            mp = [critical_frames[0]]
            ml = [-1]

            # detect critical subparts
            for cb in crit_bounds:
                # escape & entry
                mp += [critical_frames[cb], critical_frames[cb + 1]]
                ml += [-1, -1]

            # mark end of critical frames as critical
            mp += [critical_frames[-1]]
            ml += [-1]

        # merge detected
        visit_index = np.concatenate((visit_index, mp))
        visit_label = np.concatenate((visit_label, ml))

    # sort entries based on appearance (frame ids)
    sort_idx = visit_index.argsort()
    visit_index = visit_index[sort_idx]
    visit_label = visit_label[sort_idx]

    if return_plot_info:
        return plot_details
    else:
        return visit_index, visit_label


def map_trajectory(folder_name, cache=True, local_mapping=True, global_mapping=True, local_mapping_raw=True, global_transformation_type='affine', pos_jump_th=5):
    """
    applies mapping to the raw data stored in folder_name and stores them in the transformed data in folder_name

    Args:
        folder_name (str): path where raw data can be found can results will be saved
        cache (bool): if true results are stored
        local_mapping(bool): if true applies local mapping and creates "_local" files
        global_mapping(bool): if true applies global mapping and creates "_global" files

    """
    # distance tensor, landmark_ids = get distance tensor
    landmark_tensor, _ = get_landmarks_pos_tensor(folder_name, recompute=True)

    # obtain landmark configuration
    marker_path = str(Path(folder_name).parent.parent.resolve())
    marker_dict = get_marker_dict(marker_path)

    _, landmark_ids = get_landmarks(folder_name)

    traj_df = get_trajectory(folder_name, df=True)
    traj = traj_df.iloc[:, 1:4].to_numpy()

    # obtain trajectory splits whenever visiting landmarks
    sub_idx, sub_label = subtrajectory_detection(trajectory=traj, tensor=landmark_tensor, mark_critical_frames=True)

    # compute local transformation for each subtrajectory
    if local_mapping:
        local_transformation(landmark_tensor=landmark_tensor, traj_df=traj_df,
                             sub_idx=sub_idx, sub_label=sub_label,
                             landmark_ids=landmark_ids, marker_dict=marker_dict,
                             cache=cache, cache_path=folder_name,
                             pos_jump_th=pos_jump_th)

    # compute global transformation function based on median landmark positions and apply to entire trajectory
    if global_mapping:
        affine = global_transformation_type == 'affine'
        global_transformation(traj_df=traj_df, sub_traj=sub_idx,
                              cache=cache, cache_path=folder_name,
                              affine_transformation=affine)

    # only for evaluation purposes => do not use for annotation of data!
    if local_mapping_raw:
        # obtain trajectory splits whenever visiting landmarks
        sub_idx, sub_label = subtrajectory_detection(trajectory=traj, tensor=landmark_tensor, mark_critical_frames=False)

        local_transformation(landmark_tensor=landmark_tensor, traj_df=traj_df,
                             sub_idx=sub_idx, sub_label=sub_label,
                             landmark_ids=landmark_ids, marker_dict=marker_dict,
                             cache=cache, cache_path=folder_name,
                             pos_jump_th=100, raw=True)
