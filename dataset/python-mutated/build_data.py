"""Contains common utility functions and classes for building dataset.

This script contains utility functions and classes to converts dataset to
TFRecord file format with Example protos.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import collections
import six
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_enum('image_format', 'png', ['jpg', 'jpeg', 'png'], 'Image format.')
tf.app.flags.DEFINE_enum('label_format', 'png', ['png'], 'Segmentation label format.')
_IMAGE_FORMAT_MAP = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png'}

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, image_format='jpeg', channels=3):
        if False:
            i = 10
            return i + 15
        "Class constructor.\n\n    Args:\n      image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.\n      channels: Image channels.\n    "
        with tf.Graph().as_default():
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._session = tf.Session()
            if self._image_format in ('jpeg', 'jpg'):
                self._decode = tf.image.decode_jpeg(self._decode_data, channels=channels)
            elif self._image_format == 'png':
                self._decode = tf.image.decode_png(self._decode_data, channels=channels)

    def read_image_dims(self, image_data):
        if False:
            while True:
                i = 10
        'Reads the image dimensions.\n\n    Args:\n      image_data: string of image data.\n\n    Returns:\n      image_height and image_width.\n    '
        image = self.decode_image(image_data)
        return image.shape[:2]

    def decode_image(self, image_data):
        if False:
            return 10
        'Decodes the image data string.\n\n    Args:\n      image_data: string of image data.\n\n    Returns:\n      Decoded image data.\n\n    Raises:\n      ValueError: Value of image channels not supported.\n    '
        image = self._session.run(self._decode, feed_dict={self._decode_data: image_data})
        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError('The image channels not supported.')
        return image

def _int64_list_feature(values):
    if False:
        while True:
            i = 10
    'Returns a TF-Feature of int64_list.\n\n  Args:\n    values: A scalar or list of values.\n\n  Returns:\n    A TF-Feature.\n  '
    if not isinstance(values, collections.Iterable):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_list_feature(values):
    if False:
        print('Hello World!')
    'Returns a TF-Feature of bytes.\n\n  Args:\n    values: A string.\n\n  Returns:\n    A TF-Feature.\n  '

    def norm2bytes(value):
        if False:
            return 10
        return value.encode() if isinstance(value, str) and six.PY3 else value
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))

def image_seg_to_tfexample(image_data, filename, height, width, seg_data):
    if False:
        i = 10
        return i + 15
    'Converts one image/segmentation pair to tf example.\n\n  Args:\n    image_data: string of image data.\n    filename: image filename.\n    height: image height.\n    width: image width.\n    seg_data: string of semantic segmentation data.\n\n  Returns:\n    tf example of one image/segmentation pair.\n  '
    return tf.train.Example(features=tf.train.Features(feature={'image/encoded': _bytes_list_feature(image_data), 'image/filename': _bytes_list_feature(filename), 'image/format': _bytes_list_feature(_IMAGE_FORMAT_MAP[FLAGS.image_format]), 'image/height': _int64_list_feature(height), 'image/width': _int64_list_feature(width), 'image/channels': _int64_list_feature(3), 'image/segmentation/class/encoded': _bytes_list_feature(seg_data), 'image/segmentation/class/format': _bytes_list_feature(FLAGS.label_format)}))