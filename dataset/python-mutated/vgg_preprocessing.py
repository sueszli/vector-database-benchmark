"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

def _crop(image, offset_height, offset_width, crop_height, crop_width):
    if False:
        i = 10
        return i + 15
    "Crops the given image using the provided offsets and sizes.\n\n  Note that the method doesn't assume we know the input image size but it does\n  assume we know the input image rank.\n\n  Args:\n    image: an image of shape [height, width, channels].\n    offset_height: a scalar tensor indicating the height offset.\n    offset_width: a scalar tensor indicating the width offset.\n    crop_height: the height of the cropped image.\n    crop_width: the width of the cropped image.\n\n  Returns:\n    the cropped (and resized) image.\n\n  Raises:\n    InvalidArgumentError: if the rank is not 3 or if the image dimensions are\n      less than the crop size.\n  "
    original_shape = tf.shape(image)
    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])
    size_assertion = tf.Assert(tf.logical_and(tf.greater_equal(original_shape[0], crop_height), tf.greater_equal(original_shape[1], crop_width)), ['Crop size greater than the image size.'])
    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)

def _random_crop(image_list, crop_height, crop_width):
    if False:
        for i in range(10):
            print('nop')
    'Crops the given list of images.\n\n  The function applies the same crop to each image in the list. This can be\n  effectively applied when there are multiple image inputs of the same\n  dimension such as:\n\n    image, depths, normals = _random_crop([image, depths, normals], 120, 150)\n\n  Args:\n    image_list: a list of image tensors of the same dimension but possibly\n      varying channel.\n    crop_height: the new height.\n    crop_width: the new width.\n\n  Returns:\n    the image_list with cropped images.\n\n  Raises:\n    ValueError: if there are multiple image inputs provided with different size\n      or the images are smaller than the crop dimensions.\n  '
    if not image_list:
        raise ValueError('Empty image_list.')
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(tf.equal(image_rank, 3), ['Wrong rank for tensor  %s [expected] [actual]', image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)
    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(tf.logical_and(tf.greater_equal(image_height, crop_height), tf.greater_equal(image_width, crop_width)), ['Crop size greater than the image size.'])
    asserts = [rank_assertions[0], crop_size_assert]
    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]
        height_assert = tf.Assert(tf.equal(height, image_height), ['Wrong height for tensor %s [expected][actual]', image.name, height, image_height])
        width_assert = tf.Assert(tf.equal(width, image_width), ['Wrong width for tensor %s [expected][actual]', image.name, width, image_width])
        asserts.extend([height_assert, width_assert])
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    with tf.control_dependencies(asserts):
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)
    return [_crop(image, offset_height, offset_width, crop_height, crop_width) for image in image_list]

def _central_crop(image_list, crop_height, crop_width):
    if False:
        while True:
            i = 10
    'Performs central crops of the given image list.\n\n  Args:\n    image_list: a list of image tensors of the same dimension but possibly\n      varying channel.\n    crop_height: the height of the image following the crop.\n    crop_width: the width of the image following the crop.\n\n  Returns:\n    the list of cropped images.\n  '
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2
        outputs.append(_crop(image, offset_height, offset_width, crop_height, crop_width))
    return outputs

def _mean_image_subtraction(image, means):
    if False:
        while True:
            i = 10
    "Subtracts the given means from each image channel.\n\n  For example:\n    means = [123.68, 116.779, 103.939]\n    image = _mean_image_subtraction(image, means)\n\n  Note that the rank of `image` must be known.\n\n  Args:\n    image: a tensor of size [height, width, C].\n    means: a C-vector of values to subtract from each channel.\n\n  Returns:\n    the centered image.\n\n  Raises:\n    ValueError: If the rank of `image` is unknown, if `image` has a rank other\n      than three or if the number of channels in `image` doesn't match the\n      number of values in `means`.\n  "
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)

def _smallest_size_at_least(height, width, smallest_side):
    if False:
        print('Hello World!')
    'Computes new shape with the smallest side equal to `smallest_side`.\n\n  Computes new shape with the smallest side equal to `smallest_side` while\n  preserving the original aspect ratio.\n\n  Args:\n    height: an int32 scalar tensor indicating the current height.\n    width: an int32 scalar tensor indicating the current width.\n    smallest_side: A python integer or scalar `Tensor` indicating the size of\n      the smallest side after resize.\n\n  Returns:\n    new_height: an int32 scalar tensor indicating the new height.\n    new_width: and int32 scalar tensor indicating the new width.\n  '
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)
    scale = tf.cond(tf.greater(height, width), lambda : smallest_side / width, lambda : smallest_side / height)
    new_height = tf.to_int32(tf.rint(height * scale))
    new_width = tf.to_int32(tf.rint(width * scale))
    return (new_height, new_width)

def _aspect_preserving_resize(image, smallest_side):
    if False:
        i = 10
        return i + 15
    'Resize images preserving the original aspect ratio.\n\n  Args:\n    image: A 3-D image `Tensor`.\n    smallest_side: A python integer or scalar `Tensor` indicating the size of\n      the smallest side after resize.\n\n  Returns:\n    resized_image: A 3-D tensor containing the resized image.\n  '
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    (new_height, new_width) = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image

def preprocess_for_train(image, output_height, output_width, resize_side_min=_RESIZE_SIDE_MIN, resize_side_max=_RESIZE_SIDE_MAX):
    if False:
        i = 10
        return i + 15
    'Preprocesses the given image for training.\n\n  Note that the actual resizing scale is sampled from\n    [`resize_size_min`, `resize_size_max`].\n\n  Args:\n    image: A `Tensor` representing an image of arbitrary size.\n    output_height: The height of the image after preprocessing.\n    output_width: The width of the image after preprocessing.\n    resize_side_min: The lower bound for the smallest side of the image for\n      aspect-preserving resizing.\n    resize_side_max: The upper bound for the smallest side of the image for\n      aspect-preserving resizing.\n\n  Returns:\n    A preprocessed image.\n  '
    resize_side = tf.random_uniform([], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)
    image = _aspect_preserving_resize(image, resize_side)
    image = _random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

def preprocess_for_eval(image, output_height, output_width, resize_side):
    if False:
        print('Hello World!')
    'Preprocesses the given image for evaluation.\n\n  Args:\n    image: A `Tensor` representing an image of arbitrary size.\n    output_height: The height of the image after preprocessing.\n    output_width: The width of the image after preprocessing.\n    resize_side: The smallest side of the image for aspect-preserving resizing.\n\n  Returns:\n    A preprocessed image.\n  '
    image = _aspect_preserving_resize(image, resize_side)
    image = _central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

def preprocess_image(image, output_height, output_width, is_training=False, resize_side_min=_RESIZE_SIDE_MIN, resize_side_max=_RESIZE_SIDE_MAX):
    if False:
        return 10
    "Preprocesses the given image.\n\n  Args:\n    image: A `Tensor` representing an image of arbitrary size.\n    output_height: The height of the image after preprocessing.\n    output_width: The width of the image after preprocessing.\n    is_training: `True` if we're preprocessing the image for training and\n      `False` otherwise.\n    resize_side_min: The lower bound for the smallest side of the image for\n      aspect-preserving resizing. If `is_training` is `False`, then this value\n      is used for rescaling.\n    resize_side_max: The upper bound for the smallest side of the image for\n      aspect-preserving resizing. If `is_training` is `False`, this value is\n      ignored. Otherwise, the resize side is sampled from\n        [resize_size_min, resize_size_max].\n\n  Returns:\n    A preprocessed image.\n  "
    if is_training:
        return preprocess_for_train(image, output_height, output_width, resize_side_min, resize_side_max)
    else:
        return preprocess_for_eval(image, output_height, output_width, resize_side_min)