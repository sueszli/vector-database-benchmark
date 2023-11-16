"""Provides utilities for preprocessing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim

def preprocess_image(image, output_height, output_width, is_training):
    if False:
        while True:
            i = 10
    "Preprocesses the given image.\n\n  Args:\n    image: A `Tensor` representing an image of arbitrary size.\n    output_height: The height of the image after preprocessing.\n    output_width: The width of the image after preprocessing.\n    is_training: `True` if we're preprocessing the image for training and\n      `False` otherwise.\n\n  Returns:\n    A preprocessed image.\n  "
    image = tf.to_float(image)
    image = tf.image.resize_image_with_crop_or_pad(image, output_width, output_height)
    image = tf.subtract(image, 128.0)
    image = tf.div(image, 128.0)
    return image