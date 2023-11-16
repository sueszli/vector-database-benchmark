"""Bounding Box List definition.

BoxList represents a list of bounding boxes as tensorflow
tensors, where each bounding box is represented as a row of 4 numbers,
[y_min, x_min, y_max, x_max].  It is assumed that all bounding boxes
within a given list correspond to a single image.  See also
box_list_ops.py for common box related operations (such as area, iou, etc).

Optionally, users can add additional related fields (such as weights).
We assume the following things to be true about fields:
* they correspond to boxes in the box_list along the 0th dimension
* they have inferrable rank at graph construction time
* all dimensions except for possibly the 0th can be inferred
  (i.e., not None) at graph construction time.

Some other notes:
  * Following tensorflow conventions, we use height, width ordering,
  and correspondingly, y,x (or ymin, xmin, ymax, xmax) ordering
  * Tensors are always provided as (flat) [N, 4] tensors.
"""
import tensorflow.compat.v2 as tf

class BoxList(object):
    """Box collection."""

    def __init__(self, boxes):
        if False:
            i = 10
            return i + 15
        'Constructs box collection.\n\n    Args:\n      boxes: a tensor of shape [N, 4] representing box corners\n\n    Raises:\n      ValueError: if invalid dimensions for bbox data or if bbox data is not in\n          float32 format.\n    '
        if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
            raise ValueError('Invalid dimensions for box data.')
        if boxes.dtype != tf.float32:
            raise ValueError('Invalid tensor type: should be tf.float32')
        self.data = {'boxes': boxes}

    def num_boxes(self):
        if False:
            print('Hello World!')
        'Returns number of boxes held in collection.\n\n    Returns:\n      a tensor representing the number of boxes held in the collection.\n    '
        return tf.shape(input=self.data['boxes'])[0]

    def num_boxes_static(self):
        if False:
            while True:
                i = 10
        'Returns number of boxes held in collection.\n\n    This number is inferred at graph construction time rather than run-time.\n\n    Returns:\n      Number of boxes held in collection (integer) or None if this is not\n        inferrable at graph construction time.\n    '
        return self.data['boxes'].get_shape().dims[0].value

    def get_all_fields(self):
        if False:
            print('Hello World!')
        'Returns all fields.'
        return self.data.keys()

    def get_extra_fields(self):
        if False:
            print('Hello World!')
        "Returns all non-box fields (i.e., everything not named 'boxes')."
        return [k for k in self.data.keys() if k != 'boxes']

    def add_field(self, field, field_data):
        if False:
            while True:
                i = 10
        'Add field to box list.\n\n    This method can be used to add related box data such as\n    weights/labels, etc.\n\n    Args:\n      field: a string key to access the data via `get`\n      field_data: a tensor containing the data to store in the BoxList\n    '
        self.data[field] = field_data

    def has_field(self, field):
        if False:
            print('Hello World!')
        return field in self.data

    def get(self):
        if False:
            while True:
                i = 10
        'Convenience function for accessing box coordinates.\n\n    Returns:\n      a tensor with shape [N, 4] representing box coordinates.\n    '
        return self.get_field('boxes')

    def set(self, boxes):
        if False:
            while True:
                i = 10
        'Convenience function for setting box coordinates.\n\n    Args:\n      boxes: a tensor of shape [N, 4] representing box corners\n\n    Raises:\n      ValueError: if invalid dimensions for bbox data\n    '
        if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
            raise ValueError('Invalid dimensions for box data.')
        self.data['boxes'] = boxes

    def get_field(self, field):
        if False:
            while True:
                i = 10
        'Accesses a box collection and associated fields.\n\n    This function returns specified field with object; if no field is specified,\n    it returns the box coordinates.\n\n    Args:\n      field: this optional string parameter can be used to specify\n        a related field to be accessed.\n\n    Returns:\n      a tensor representing the box collection or an associated field.\n\n    Raises:\n      ValueError: if invalid field\n    '
        if not self.has_field(field):
            raise ValueError('field ' + str(field) + ' does not exist')
        return self.data[field]

    def set_field(self, field, value):
        if False:
            i = 10
            return i + 15
        'Sets the value of a field.\n\n    Updates the field of a box_list with a given value.\n\n    Args:\n      field: (string) name of the field to set value.\n      value: the value to assign to the field.\n\n    Raises:\n      ValueError: if the box_list does not have specified field.\n    '
        if not self.has_field(field):
            raise ValueError('field %s does not exist' % field)
        self.data[field] = value

    def get_center_coordinates_and_sizes(self, scope=None):
        if False:
            return 10
        'Computes the center coordinates, height and width of the boxes.\n\n    Args:\n      scope: name scope of the function.\n\n    Returns:\n      a list of 4 1-D tensors [ycenter, xcenter, height, width].\n    '
        if not scope:
            scope = 'get_center_coordinates_and_sizes'
        with tf.name_scope(scope):
            box_corners = self.get()
            (ymin, xmin, ymax, xmax) = tf.unstack(tf.transpose(a=box_corners))
            width = xmax - xmin
            height = ymax - ymin
            ycenter = ymin + height / 2.0
            xcenter = xmin + width / 2.0
            return [ycenter, xcenter, height, width]

    def transpose_coordinates(self, scope=None):
        if False:
            while True:
                i = 10
        'Transpose the coordinate representation in a boxlist.\n\n    Args:\n      scope: name scope of the function.\n    '
        if not scope:
            scope = 'transpose_coordinates'
        with tf.name_scope(scope):
            (y_min, x_min, y_max, x_max) = tf.split(value=self.get(), num_or_size_splits=4, axis=1)
            self.set(tf.concat([x_min, y_min, x_max, y_max], 1))

    def as_tensor_dict(self, fields=None):
        if False:
            while True:
                i = 10
        'Retrieves specified fields as a dictionary of tensors.\n\n    Args:\n      fields: (optional) list of fields to return in the dictionary.\n        If None (default), all fields are returned.\n\n    Returns:\n      tensor_dict: A dictionary of tensors specified by fields.\n\n    Raises:\n      ValueError: if specified field is not contained in boxlist.\n    '
        tensor_dict = {}
        if fields is None:
            fields = self.get_all_fields()
        for field in fields:
            if not self.has_field(field):
                raise ValueError('boxlist must contain all specified fields')
            tensor_dict[field] = self.get_field(field)
        return tensor_dict