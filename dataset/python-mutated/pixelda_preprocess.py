"""Contains functions for preprocessing the inputs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def preprocess_classification(image, labels, is_training=False):
    if False:
        return 10
    "Preprocesses the image and labels for classification purposes.\n\n  Preprocessing includes shifting the images to be 0-centered between -1 and 1.\n  This is not only a popular method of preprocessing (inception) but is also\n  the mechanism used by DSNs.\n\n  Args:\n    image: A `Tensor` of size [height, width, 3].\n    labels: A dictionary of labels.\n    is_training: Whether or not we're training the model.\n\n  Returns:\n    The preprocessed image and labels.\n  "
    image = tf.image.convert_image_dtype(image, tf.float32)
    image -= 0.5
    image *= 2
    return (image, labels)

def preprocess_style_transfer(image, labels, augment=False, size=None, is_training=False):
    if False:
        print('Hello World!')
    "Preprocesses the image and labels for style transfer purposes.\n\n  Args:\n    image: A `Tensor` of size [height, width, 3].\n    labels: A dictionary of labels.\n    augment: Whether to apply data augmentation to inputs\n    size: The height and width to which images should be resized. If left as\n      `None`, then no resizing is performed\n    is_training: Whether or not we're training the model\n\n  Returns:\n    The preprocessed image and labels. Scaled to [-1, 1]\n  "
    image = tf.image.convert_image_dtype(image, tf.float32)
    if augment and is_training:
        image = image_augmentation(image)
    if size:
        image = resize_image(image, size)
    image -= 0.5
    image *= 2
    return (image, labels)

def image_augmentation(image):
    if False:
        print('Hello World!')
    'Performs data augmentation by randomly permuting the inputs.\n\n  Args:\n    image: A float `Tensor` of size [height, width, channels] with values\n      in range[0,1].\n\n  Returns:\n    The mutated batch of images\n  '
    num_channels = image.shape_as_list()[-1]
    if num_channels == 4:
        (image, depth) = (image[:, :, 0:3], image[:, :, 3:4])
    elif num_channels == 1:
        image = tf.image.grayscale_to_rgb(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.032)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0, 1.0)
    if num_channels == 4:
        image = tf.concat(2, [image, depth])
    elif num_channels == 1:
        image = tf.image.rgb_to_grayscale(image)
    return image

def resize_image(image, size=None):
    if False:
        return 10
    'Resize image to target size.\n\n  Args:\n    image: A `Tensor` of size [height, width, 3].\n    size: (height, width) to resize image to.\n\n  Returns:\n    resized image\n  '
    if size is None:
        raise ValueError('Must specify size')
    if image.shape_as_list()[:2] == size:
        return image
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_images(image, size)
    image = tf.squeeze(image, 0)
    return image