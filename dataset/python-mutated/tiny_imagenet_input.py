"""Tiny imagenet input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import flags
import tensorflow as tf
FLAGS = flags.FLAGS
flags.DEFINE_string('tiny_imagenet_data_dir', None, 'Directory with Tiny Imagenet dataset in TFRecord format.')

def tiny_imagenet_parser(value, image_size, is_training):
    if False:
        while True:
            i = 10
    'Parses tiny imagenet example.\n\n  Args:\n    value: encoded example.\n    image_size: size of the image.\n    is_training: if True then do training preprocessing (which includes\n      random cropping), otherwise do eval preprocessing.\n\n  Returns:\n    image: tensor with the image.\n    label: true label of the image.\n  '
    keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, ''), 'label/tiny_imagenet': tf.FixedLenFeature([], tf.int64, -1)}
    parsed = tf.parse_single_example(value, keys_to_features)
    image_buffer = tf.reshape(parsed['image/encoded'], shape=[])
    image = tf.image.decode_image(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_training:
        (bbox_begin, bbox_size, _) = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]), min_object_covered=0.5, aspect_ratio_range=[0.75, 1.33], area_range=[0.5, 1.0], max_attempts=20, use_image_if_no_bounding_boxes=True)
        image = tf.slice(image, bbox_begin, bbox_size)
    image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
    image = tf.multiply(tf.subtract(image, 0.5), 2.0)
    image = tf.reshape(image, [image_size, image_size, 3])
    label = tf.cast(tf.reshape(parsed['label/tiny_imagenet'], shape=[]), dtype=tf.int32)
    return (image, label)

def tiny_imagenet_input(split, batch_size, image_size, is_training):
    if False:
        return 10
    'Returns Tiny Imagenet Dataset.\n\n  Args:\n    split: name of the split, "train" or "validation".\n    batch_size: size of the minibatch.\n    image_size: size of the one side of the image. Output images will be\n      resized to square shape image_size*image_size.\n    is_training: if True then training preprocessing is done, otherwise eval\n      preprocessing is done.instance of tf.data.Dataset with the dataset.\n\n  Raises:\n    ValueError: if name of the split is incorrect.\n\n  Returns:\n    Instance of tf.data.Dataset with the dataset.\n  '
    if split.lower().startswith('train'):
        filepath = os.path.join(FLAGS.tiny_imagenet_data_dir, 'train.tfrecord')
    elif split.lower().startswith('validation'):
        filepath = os.path.join(FLAGS.tiny_imagenet_data_dir, 'validation.tfrecord')
    else:
        raise ValueError('Invalid split: %s' % split)
    dataset = tf.data.TFRecordDataset(filepath, buffer_size=8 * 1024 * 1024)
    if is_training:
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()
    dataset = dataset.apply(tf.data.experimental.map_and_batch(lambda value: tiny_imagenet_parser(value, image_size, is_training), batch_size=batch_size, num_parallel_batches=4, drop_remainder=True))

    def set_shapes(images, labels):
        if False:
            for i in range(10):
                print('nop')
        'Statically set the batch_size dimension.'
        images.set_shape(images.get_shape().merge_with(tf.TensorShape([batch_size, None, None, None])))
        labels.set_shape(labels.get_shape().merge_with(tf.TensorShape([batch_size])))
        return (images, labels)
    dataset = dataset.map(set_shapes)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def num_examples_per_epoch(split):
    if False:
        while True:
            i = 10
    'Returns the number of examples in the data set.\n\n  Args:\n    split: name of the split, "train" or "validation".\n\n  Raises:\n    ValueError: if split name is incorrect.\n\n  Returns:\n    Number of example in the split.\n  '
    if split.lower().startswith('train'):
        return 100000
    elif split.lower().startswith('validation'):
        return 10000
    else:
        raise ValueError('Invalid split: %s' % split)