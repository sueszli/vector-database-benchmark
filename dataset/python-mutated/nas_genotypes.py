"""Genotypes used by NAS."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib import slim as contrib_slim
from deeplab.core import nas_cell
slim = contrib_slim

class PNASCell(nas_cell.NASBaseCell):
    """Configuration and construction of the PNASNet-5 Cell."""

    def __init__(self, num_conv_filters, drop_path_keep_prob, total_num_cells, total_training_steps, batch_norm_fn=slim.batch_norm):
        if False:
            while True:
                i = 10
        operations = ['separable_5x5_2', 'max_pool_3x3', 'separable_7x7_2', 'max_pool_3x3', 'separable_5x5_2', 'separable_3x3_2', 'separable_3x3_2', 'max_pool_3x3', 'separable_3x3_2', 'none']
        used_hiddenstates = [1, 1, 0, 0, 0, 0, 0]
        hiddenstate_indices = [1, 1, 0, 0, 0, 0, 4, 0, 1, 0]
        super(PNASCell, self).__init__(num_conv_filters, operations, used_hiddenstates, hiddenstate_indices, drop_path_keep_prob, total_num_cells, total_training_steps, batch_norm_fn)