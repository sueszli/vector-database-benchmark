"""Provides utilities to Cifar-10 dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import logging
import tensorflow as tf
from official.vision.image_classification import imagenet_preprocessing
HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
NUM_IMAGES = {'train': 50000, 'validation': 10000}
_NUM_DATA_FILES = 5
NUM_CLASSES = 10

def parse_record(raw_record, is_training, dtype):
    if False:
        while True:
            i = 10
    'Parses a record containing a training example of an image.\n\n  The input record is parsed into a label and image, and the image is passed\n  through preprocessing steps (cropping, flipping, and so on).\n\n  This method converts the label to one hot to fit the loss function.\n\n  Args:\n    raw_record: scalar Tensor tf.string containing a serialized\n      Example protocol buffer.\n    is_training: A boolean denoting whether the input is for training.\n    dtype: Data type to use for input images.\n\n  Returns:\n    Tuple with processed image tensor and one-hot-encoded label tensor.\n  '
    record_vector = tf.io.decode_raw(raw_record, tf.uint8)
    label = tf.cast(record_vector[0], tf.int32)
    depth_major = tf.reshape(record_vector[1:_RECORD_BYTES], [NUM_CHANNELS, HEIGHT, WIDTH])
    image = tf.cast(tf.transpose(a=depth_major, perm=[1, 2, 0]), tf.float32)
    image = preprocess_image(image, is_training)
    image = tf.cast(image, dtype)
    label = tf.compat.v1.sparse_to_dense(label, (NUM_CLASSES,), 1)
    return (image, label)

def preprocess_image(image, is_training):
    if False:
        return 10
    'Preprocess a single image of layout [height, width, depth].'
    if is_training:
        image = tf.image.resize_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)
        image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])
        image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    return image

def get_filenames(is_training, data_dir):
    if False:
        print('Hello World!')
    'Returns a list of filenames.'
    assert tf.io.gfile.exists(data_dir), 'Run cifar10_download_and_extract.py first to download and extract the CIFAR-10 data.'
    if is_training:
        return [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, _NUM_DATA_FILES + 1)]
    else:
        return [os.path.join(data_dir, 'test_batch.bin')]

def input_fn(is_training, data_dir, batch_size, num_epochs=1, dtype=tf.float32, datasets_num_private_threads=None, parse_record_fn=parse_record, input_context=None, drop_remainder=False):
    if False:
        print('Hello World!')
    'Input function which provides batches for train or eval.\n\n  Args:\n    is_training: A boolean denoting whether the input is for training.\n    data_dir: The directory containing the input data.\n    batch_size: The number of samples per batch.\n    num_epochs: The number of epochs to repeat the dataset.\n    dtype: Data type to use for images/features\n    datasets_num_private_threads: Number of private threads for tf.data.\n    parse_record_fn: Function to use for parsing the records.\n    input_context: A `tf.distribute.InputContext` object passed in by\n      `tf.distribute.Strategy`.\n    drop_remainder: A boolean indicates whether to drop the remainder of the\n      batches. If True, the batch dimension will be static.\n\n  Returns:\n    A dataset that can be used for iteration.\n  '
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
    if input_context:
        logging.info('Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d', input_context.input_pipeline_id, input_context.num_input_pipelines)
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    return imagenet_preprocessing.process_record_dataset(dataset=dataset, is_training=is_training, batch_size=batch_size, shuffle_buffer=NUM_IMAGES['train'], parse_record_fn=parse_record_fn, num_epochs=num_epochs, dtype=dtype, datasets_num_private_threads=datasets_num_private_threads, drop_remainder=drop_remainder)