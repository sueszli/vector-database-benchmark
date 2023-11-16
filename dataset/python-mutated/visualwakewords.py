"""Provides data for Visual WakeWords Dataset with images+labels.

Visual WakeWords Dataset derives from the COCO dataset to design tiny models
classifying two classes, such as person/not-person. The COCO annotations
are filtered to two classes: person and not-person (or another user-defined
category). Bounding boxes for small objects with area less than 5% of the image
area are filtered out.
See build_visualwakewords_data.py which generates the Visual WakeWords dataset
annotations from the raw COCO dataset and converts them to TFRecord.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from datasets import dataset_utils
slim = contrib_slim
_FILE_PATTERN = '%s.record-*'
_SPLITS_TO_SIZES = {'train': 82783, 'val': 40504}
_ITEMS_TO_DESCRIPTIONS = {'image': 'A color image of varying height and width.', 'label': 'The label id of the image, an integer in {0, 1}', 'object/bbox': 'A list of bounding boxes.'}
_NUM_CLASSES = 2
LABELS_FILENAME = 'labels.txt'

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    if False:
        return 10
    "Gets a dataset tuple with instructions for reading ImageNet.\n\n  Args:\n    split_name: A train/test split name.\n    dataset_dir: The base directory of the dataset sources.\n    file_pattern: The file pattern to use when matching the dataset sources. It\n      is assumed that the pattern contains a '%s' string so that the split name\n      can be inserted.\n    reader: The TensorFlow reader type.\n\n  Returns:\n    A `Dataset` namedtuple.\n\n  Raises:\n    ValueError: if `split_name` is not a valid train/test split.\n  "
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''), 'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'), 'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1), 'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32), 'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32), 'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32), 'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)}
    items_to_handlers = {'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'), 'label': slim.tfexample_decoder.Tensor('image/class/label'), 'object/bbox': slim.tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/')}
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    labels_to_names = None
    labels_file = os.path.join(dataset_dir, LABELS_FILENAME)
    if tf.gfile.Exists(labels_file):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
    return slim.dataset.Dataset(data_sources=file_pattern, reader=reader, decoder=decoder, num_samples=_SPLITS_TO_SIZES[split_name], items_to_descriptions=_ITEMS_TO_DESCRIPTIONS, num_classes=_NUM_CLASSES, labels_to_names=labels_to_names)