"""Numpy BoxMaskList classes and functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from object_detection.utils import np_box_list

class BoxMaskList(np_box_list.BoxList):
    """Convenience wrapper for BoxList with masks.

  BoxMaskList extends the np_box_list.BoxList to contain masks as well.
  In particular, its constructor receives both boxes and masks. Note that the
  masks correspond to the full image.
  """

    def __init__(self, box_data, mask_data):
        if False:
            for i in range(10):
                print('nop')
        'Constructs box collection.\n\n    Args:\n      box_data: a numpy array of shape [N, 4] representing box coordinates\n      mask_data: a numpy array of shape [N, height, width] representing masks\n        with values are in {0,1}. The masks correspond to the full\n        image. The height and the width will be equal to image height and width.\n\n    Raises:\n      ValueError: if bbox data is not a numpy array\n      ValueError: if invalid dimensions for bbox data\n      ValueError: if mask data is not a numpy array\n      ValueError: if invalid dimension for mask data\n    '
        super(BoxMaskList, self).__init__(box_data)
        if not isinstance(mask_data, np.ndarray):
            raise ValueError('Mask data must be a numpy array.')
        if len(mask_data.shape) != 3:
            raise ValueError('Invalid dimensions for mask data.')
        if mask_data.dtype != np.uint8:
            raise ValueError('Invalid data type for mask data: uint8 is required.')
        if mask_data.shape[0] != box_data.shape[0]:
            raise ValueError('There should be the same number of boxes and masks.')
        self.data['masks'] = mask_data

    def get_masks(self):
        if False:
            return 10
        'Convenience function for accessing masks.\n\n    Returns:\n      a numpy array of shape [N, height, width] representing masks\n    '
        return self.get_field('masks')