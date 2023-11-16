"""3D->2D projector model as used in PTN (NIPS16)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from nets import perspective_transform

def model(voxels, transform_matrix, params, is_training):
    if False:
        while True:
            i = 10
    'Model transforming the 3D voxels into 2D projections.\n\n  Args:\n    voxels: A tensor of size [batch, depth, height, width, channel]\n      representing the input of projection layer (tf.float32).\n    transform_matrix: A tensor of size [batch, 16] representing\n      the flattened 4-by-4 matrix for transformation (tf.float32).\n    params: Model parameters (dict).\n    is_training: Set to True if while training (boolean).\n\n  Returns:\n    A transformed tensor (tf.float32)\n\n  '
    del is_training
    voxels = tf.transpose(voxels, [0, 2, 1, 3, 4])
    z_near = params.focal_length
    z_far = params.focal_length + params.focal_range
    transformed_voxels = perspective_transform.transformer(voxels, transform_matrix, [params.vox_size] * 3, z_near, z_far)
    views = tf.reduce_max(transformed_voxels, [1])
    views = tf.reverse(views, [1])
    return views