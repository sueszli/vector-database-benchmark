"""Keypoint operations.

Keypoints are represented as tensors of shape [num_instances, num_keypoints, 2],
where the last dimension holds rank 2 tensors of the form [y, x] representing
the coordinates of the keypoint.
"""
import numpy as np
import tensorflow as tf

def scale(keypoints, y_scale, x_scale, scope=None):
    if False:
        print('Hello World!')
    'Scales keypoint coordinates in x and y dimensions.\n\n  Args:\n    keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n    y_scale: (float) scalar tensor\n    x_scale: (float) scalar tensor\n    scope: name scope.\n\n  Returns:\n    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n  '
    with tf.name_scope(scope, 'Scale'):
        y_scale = tf.cast(y_scale, tf.float32)
        x_scale = tf.cast(x_scale, tf.float32)
        new_keypoints = keypoints * [[[y_scale, x_scale]]]
        return new_keypoints

def clip_to_window(keypoints, window, scope=None):
    if False:
        while True:
            i = 10
    'Clips keypoints to a window.\n\n  This op clips any input keypoints to a window.\n\n  Args:\n    keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]\n      window to which the op should clip the keypoints.\n    scope: name scope.\n\n  Returns:\n    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n  '
    with tf.name_scope(scope, 'ClipToWindow'):
        (y, x) = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
        (win_y_min, win_x_min, win_y_max, win_x_max) = tf.unstack(window)
        y = tf.maximum(tf.minimum(y, win_y_max), win_y_min)
        x = tf.maximum(tf.minimum(x, win_x_max), win_x_min)
        new_keypoints = tf.concat([y, x], 2)
        return new_keypoints

def prune_outside_window(keypoints, window, scope=None):
    if False:
        print('Hello World!')
    'Prunes keypoints that fall outside a given window.\n\n  This function replaces keypoints that fall outside the given window with nan.\n  See also clip_to_window which clips any keypoints that fall outside the given\n  window.\n\n  Args:\n    keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]\n      window outside of which the op should prune the keypoints.\n    scope: name scope.\n\n  Returns:\n    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n  '
    with tf.name_scope(scope, 'PruneOutsideWindow'):
        (y, x) = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
        (win_y_min, win_x_min, win_y_max, win_x_max) = tf.unstack(window)
        valid_indices = tf.logical_and(tf.logical_and(y >= win_y_min, y <= win_y_max), tf.logical_and(x >= win_x_min, x <= win_x_max))
        new_y = tf.where(valid_indices, y, np.nan * tf.ones_like(y))
        new_x = tf.where(valid_indices, x, np.nan * tf.ones_like(x))
        new_keypoints = tf.concat([new_y, new_x], 2)
        return new_keypoints

def change_coordinate_frame(keypoints, window, scope=None):
    if False:
        return 10
    "Changes coordinate frame of the keypoints to be relative to window's frame.\n\n  Given a window of the form [y_min, x_min, y_max, x_max], changes keypoint\n  coordinates from keypoints of shape [num_instances, num_keypoints, 2]\n  to be relative to this window.\n\n  An example use case is data augmentation: where we are given groundtruth\n  keypoints and would like to randomly crop the image to some window. In this\n  case we need to change the coordinate frame of each groundtruth keypoint to be\n  relative to this new window.\n\n  Args:\n    keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]\n      window we should change the coordinate frame to.\n    scope: name scope.\n\n  Returns:\n    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n  "
    with tf.name_scope(scope, 'ChangeCoordinateFrame'):
        win_height = window[2] - window[0]
        win_width = window[3] - window[1]
        new_keypoints = scale(keypoints - [window[0], window[1]], 1.0 / win_height, 1.0 / win_width)
        return new_keypoints

def to_normalized_coordinates(keypoints, height, width, check_range=True, scope=None):
    if False:
        return 10
    'Converts absolute keypoint coordinates to normalized coordinates in [0, 1].\n\n  Usually one uses the dynamic shape of the image or conv-layer tensor:\n    keypoints = keypoint_ops.to_normalized_coordinates(keypoints,\n                                                       tf.shape(images)[1],\n                                                       tf.shape(images)[2]),\n\n  This function raises an assertion failed error at graph execution time when\n  the maximum coordinate is smaller than 1.01 (which means that coordinates are\n  already normalized). The value 1.01 is to deal with small rounding errors.\n\n  Args:\n    keypoints: A tensor of shape [num_instances, num_keypoints, 2].\n    height: Maximum value for y coordinate of absolute keypoint coordinates.\n    width: Maximum value for x coordinate of absolute keypoint coordinates.\n    check_range: If True, checks if the coordinates are normalized.\n    scope: name scope.\n\n  Returns:\n    tensor of shape [num_instances, num_keypoints, 2] with normalized\n    coordinates in [0, 1].\n  '
    with tf.name_scope(scope, 'ToNormalizedCoordinates'):
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)
        if check_range:
            max_val = tf.reduce_max(keypoints)
            max_assert = tf.Assert(tf.greater(max_val, 1.01), ['max value is lower than 1.01: ', max_val])
            with tf.control_dependencies([max_assert]):
                width = tf.identity(width)
        return scale(keypoints, 1.0 / height, 1.0 / width)

def to_absolute_coordinates(keypoints, height, width, check_range=True, scope=None):
    if False:
        return 10
    'Converts normalized keypoint coordinates to absolute pixel coordinates.\n\n  This function raises an assertion failed error when the maximum keypoint\n  coordinate value is larger than 1.01 (in which case coordinates are already\n  absolute).\n\n  Args:\n    keypoints: A tensor of shape [num_instances, num_keypoints, 2]\n    height: Maximum value for y coordinate of absolute keypoint coordinates.\n    width: Maximum value for x coordinate of absolute keypoint coordinates.\n    check_range: If True, checks if the coordinates are normalized or not.\n    scope: name scope.\n\n  Returns:\n    tensor of shape [num_instances, num_keypoints, 2] with absolute coordinates\n    in terms of the image size.\n\n  '
    with tf.name_scope(scope, 'ToAbsoluteCoordinates'):
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)
        if check_range:
            max_val = tf.reduce_max(keypoints)
            max_assert = tf.Assert(tf.greater_equal(1.01, max_val), ['maximum keypoint coordinate value is larger than 1.01: ', max_val])
            with tf.control_dependencies([max_assert]):
                width = tf.identity(width)
        return scale(keypoints, height, width)

def flip_horizontal(keypoints, flip_point, flip_permutation, scope=None):
    if False:
        while True:
            i = 10
    "Flips the keypoints horizontally around the flip_point.\n\n  This operation flips the x coordinate for each keypoint around the flip_point\n  and also permutes the keypoints in a manner specified by flip_permutation.\n\n  Args:\n    keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n    flip_point:  (float) scalar tensor representing the x coordinate to flip the\n      keypoints around.\n    flip_permutation: rank 1 int32 tensor containing the keypoint flip\n      permutation. This specifies the mapping from original keypoint indices\n      to the flipped keypoint indices. This is used primarily for keypoints\n      that are not reflection invariant. E.g. Suppose there are 3 keypoints\n      representing ['head', 'right_eye', 'left_eye'], then a logical choice for\n      flip_permutation might be [0, 2, 1] since we want to swap the 'left_eye'\n      and 'right_eye' after a horizontal flip.\n    scope: name scope.\n\n  Returns:\n    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n  "
    with tf.name_scope(scope, 'FlipHorizontal'):
        keypoints = tf.transpose(keypoints, [1, 0, 2])
        keypoints = tf.gather(keypoints, flip_permutation)
        (v, u) = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
        u = flip_point * 2.0 - u
        new_keypoints = tf.concat([v, u], 2)
        new_keypoints = tf.transpose(new_keypoints, [1, 0, 2])
        return new_keypoints

def flip_vertical(keypoints, flip_point, flip_permutation, scope=None):
    if False:
        return 10
    "Flips the keypoints vertically around the flip_point.\n\n  This operation flips the y coordinate for each keypoint around the flip_point\n  and also permutes the keypoints in a manner specified by flip_permutation.\n\n  Args:\n    keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n    flip_point:  (float) scalar tensor representing the y coordinate to flip the\n      keypoints around.\n    flip_permutation: rank 1 int32 tensor containing the keypoint flip\n      permutation. This specifies the mapping from original keypoint indices\n      to the flipped keypoint indices. This is used primarily for keypoints\n      that are not reflection invariant. E.g. Suppose there are 3 keypoints\n      representing ['head', 'right_eye', 'left_eye'], then a logical choice for\n      flip_permutation might be [0, 2, 1] since we want to swap the 'left_eye'\n      and 'right_eye' after a horizontal flip.\n    scope: name scope.\n\n  Returns:\n    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n  "
    with tf.name_scope(scope, 'FlipVertical'):
        keypoints = tf.transpose(keypoints, [1, 0, 2])
        keypoints = tf.gather(keypoints, flip_permutation)
        (v, u) = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
        v = flip_point * 2.0 - v
        new_keypoints = tf.concat([v, u], 2)
        new_keypoints = tf.transpose(new_keypoints, [1, 0, 2])
        return new_keypoints

def rot90(keypoints, scope=None):
    if False:
        while True:
            i = 10
    'Rotates the keypoints counter-clockwise by 90 degrees.\n\n  Args:\n    keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n    scope: name scope.\n\n  Returns:\n    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]\n  '
    with tf.name_scope(scope, 'Rot90'):
        keypoints = tf.transpose(keypoints, [1, 0, 2])
        (v, u) = tf.split(value=keypoints[:, :, ::-1], num_or_size_splits=2, axis=2)
        v = 1.0 - v
        new_keypoints = tf.concat([v, u], 2)
        new_keypoints = tf.transpose(new_keypoints, [1, 0, 2])
        return new_keypoints