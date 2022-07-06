import numpy as np


def angle_between(vec_a, vec_b):
    """calculates the signed angle between two vectors
    Args:
        vec_a (list or 1d array): vector containing (x1,y1)
        vec_b (list or 1d array): vector containing (x2,y2)
    Returns:
        signed_angle: signed angle between the two given vectors
    """
    signed_angle = np.arctan2(vec_b[1], vec_b[0]) - np.arctan2(vec_a[1], vec_a[0])

    return signed_angle
