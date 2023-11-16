"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_flowers.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from datasets import dataset_utils
slim = contrib_slim
_FILE_PATTERN = 'flowers_%s_*.tfrecord'
SPLITS_TO_SIZES = {'train': 3320, 'validation': 350}
_NUM_CLASSES = 5
_ITEMS_TO_DESCRIPTIONS = {'image': 'A color image of varying size.', 'label': 'A single integer between 0 and 4'}

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    if False:
        i = 10
        return i + 15
    "Gets a dataset tuple with instructions for reading flowers.\n\n  Args:\n    split_name: A train/validation split name.\n    dataset_dir: The base directory of the dataset sources.\n    file_pattern: The file pattern to use when matching the dataset sources.\n      It is assumed that the pattern contains a '%s' string so that the split\n      name can be inserted.\n    reader: The TensorFlow reader type.\n\n  Returns:\n    A `Dataset` namedtuple.\n\n  Raises:\n    ValueError: if `split_name` is not a valid train/validation split.\n  "
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''), 'image/format': tf.FixedLenFeature((), tf.string, default_value='png'), 'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))}
    items_to_handlers = {'image': slim.tfexample_decoder.Image(), 'label': slim.tfexample_decoder.Tensor('image/class/label')}
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
    return slim.dataset.Dataset(data_sources=file_pattern, reader=reader, decoder=decoder, num_samples=SPLITS_TO_SIZES[split_name], items_to_descriptions=_ITEMS_TO_DESCRIPTIONS, num_classes=_NUM_CLASSES, labels_to_names=labels_to_names)