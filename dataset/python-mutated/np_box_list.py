"""Numpy BoxList classes and functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range

class BoxList(object):
    """Box collection.

  BoxList represents a list of bounding boxes as numpy array, where each
  bounding box is represented as a row of 4 numbers,
  [y_min, x_min, y_max, x_max].  It is assumed that all bounding boxes within a
  given list correspond to a single image.

  Optionally, users can add additional related fields (such as
  objectness/classification scores).
  """

    def __init__(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Constructs box collection.\n\n    Args:\n      data: a numpy array of shape [N, 4] representing box coordinates\n\n    Raises:\n      ValueError: if bbox data is not a numpy array\n      ValueError: if invalid dimensions for bbox data\n    '
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be a numpy array.')
        if len(data.shape) != 2 or data.shape[1] != 4:
            raise ValueError('Invalid dimensions for box data.')
        if data.dtype != np.float32 and data.dtype != np.float64:
            raise ValueError('Invalid data type for box data: float is required.')
        if not self._is_valid_boxes(data):
            raise ValueError('Invalid box data. data must be a numpy array of N*[y_min, x_min, y_max, x_max]')
        self.data = {'boxes': data}

    def num_boxes(self):
        if False:
            for i in range(10):
                print('nop')
        'Return number of boxes held in collections.'
        return self.data['boxes'].shape[0]

    def get_extra_fields(self):
        if False:
            print('Hello World!')
        'Return all non-box fields.'
        return [k for k in self.data.keys() if k != 'boxes']

    def has_field(self, field):
        if False:
            i = 10
            return i + 15
        return field in self.data

    def add_field(self, field, field_data):
        if False:
            for i in range(10):
                print('nop')
        'Add data to a specified field.\n\n    Args:\n      field: a string parameter used to speficy a related field to be accessed.\n      field_data: a numpy array of [N, ...] representing the data associated\n          with the field.\n    Raises:\n      ValueError: if the field is already exist or the dimension of the field\n          data does not matches the number of boxes.\n    '
        if self.has_field(field):
            raise ValueError('Field ' + field + 'already exists')
        if len(field_data.shape) < 1 or field_data.shape[0] != self.num_boxes():
            raise ValueError('Invalid dimensions for field data')
        self.data[field] = field_data

    def get(self):
        if False:
            i = 10
            return i + 15
        'Convenience function for accesssing box coordinates.\n\n    Returns:\n      a numpy array of shape [N, 4] representing box corners\n    '
        return self.get_field('boxes')

    def get_field(self, field):
        if False:
            while True:
                i = 10
        'Accesses data associated with the specified field in the box collection.\n\n    Args:\n      field: a string parameter used to speficy a related field to be accessed.\n\n    Returns:\n      a numpy 1-d array representing data of an associated field\n\n    Raises:\n      ValueError: if invalid field\n    '
        if not self.has_field(field):
            raise ValueError('field {} does not exist'.format(field))
        return self.data[field]

    def get_coordinates(self):
        if False:
            return 10
        'Get corner coordinates of boxes.\n\n    Returns:\n     a list of 4 1-d numpy arrays [y_min, x_min, y_max, x_max]\n    '
        box_coordinates = self.get()
        y_min = box_coordinates[:, 0]
        x_min = box_coordinates[:, 1]
        y_max = box_coordinates[:, 2]
        x_max = box_coordinates[:, 3]
        return [y_min, x_min, y_max, x_max]

    def _is_valid_boxes(self, data):
        if False:
            return 10
        'Check whether data fullfills the format of N*[ymin, xmin, ymax, xmin].\n\n    Args:\n      data: a numpy array of shape [N, 4] representing box coordinates\n\n    Returns:\n      a boolean indicating whether all ymax of boxes are equal or greater than\n          ymin, and all xmax of boxes are equal or greater than xmin.\n    '
        if data.shape[0] > 0:
            for i in range(data.shape[0]):
                if data[i, 0] > data[i, 2] or data[i, 1] > data[i, 3]:
                    return False
        return True