"""Provides utilities to preprocess images in CIFAR-10.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
_PADDING = 4
slim = tf.contrib.slim

def preprocess_for_train(image, output_height, output_width, padding=_PADDING, add_image_summaries=True):
    if False:
        print('Hello World!')
    'Preprocesses the given image for training.\n\n  Note that the actual resizing scale is sampled from\n    [`resize_size_min`, `resize_size_max`].\n\n  Args:\n    image: A `Tensor` representing an image of arbitrary size.\n    output_height: The height of the image after preprocessing.\n    output_width: The width of the image after preprocessing.\n    padding: The amound of padding before and after each dimension of the image.\n    add_image_summaries: Enable image summaries.\n\n  Returns:\n    A preprocessed image.\n  '
    if add_image_summaries:
        tf.summary.image('image', tf.expand_dims(image, 0))
    image = tf.to_float(image)
    if padding > 0:
        image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]])
    distorted_image = tf.random_crop(image, [output_height, output_width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    if add_image_summaries:
        tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    return tf.image.per_image_standardization(distorted_image)

def preprocess_for_eval(image, output_height, output_width, add_image_summaries=True):
    if False:
        print('Hello World!')
    'Preprocesses the given image for evaluation.\n\n  Args:\n    image: A `Tensor` representing an image of arbitrary size.\n    output_height: The height of the image after preprocessing.\n    output_width: The width of the image after preprocessing.\n    add_image_summaries: Enable image summaries.\n\n  Returns:\n    A preprocessed image.\n  '
    if add_image_summaries:
        tf.summary.image('image', tf.expand_dims(image, 0))
    image = tf.to_float(image)
    resized_image = tf.image.resize_image_with_crop_or_pad(image, output_width, output_height)
    if add_image_summaries:
        tf.summary.image('resized_image', tf.expand_dims(resized_image, 0))
    return tf.image.per_image_standardization(resized_image)

def preprocess_image(image, output_height, output_width, is_training=False, add_image_summaries=True):
    if False:
        while True:
            i = 10
    "Preprocesses the given image.\n\n  Args:\n    image: A `Tensor` representing an image of arbitrary size.\n    output_height: The height of the image after preprocessing.\n    output_width: The width of the image after preprocessing.\n    is_training: `True` if we're preprocessing the image for training and\n      `False` otherwise.\n    add_image_summaries: Enable image summaries.\n\n  Returns:\n    A preprocessed image.\n  "
    if is_training:
        return preprocess_for_train(image, output_height, output_width, add_image_summaries=add_image_summaries)
    else:
        return preprocess_for_eval(image, output_height, output_width, add_image_summaries=add_image_summaries)