"""Provides utilities to preprocess images.

Training images are sampled using the provided bounding boxes, and subsequently
cropped to the sampled bounding box. Images are additionally flipped randomly,
then resized to the target output size (without aspect-ratio preservation).

Images used during evaluation are resized (with aspect-ratio preservation) and
centrally cropped.

All images undergo mean color subtraction.

Note that these steps are colloquially referred to as "ResNet preprocessing,"
and they differ from "VGG preprocessing," which does not use bounding boxes
and instead does an aspect-preserving resize followed by random crop during
training. (These both differ from "Inception preprocessing," which introduces
color distortion steps.)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_RESIZE_MIN = 256

def _decode_crop_and_flip(image_buffer, bbox, num_channels):
    if False:
        print('Hello World!')
    'Crops the given image to a random part of the image, and randomly flips.\n\n  We use the fused decode_and_crop op, which performs better than the two ops\n  used separately in series, but note that this requires that the image be\n  passed in as an un-decoded string Tensor.\n\n  Args:\n    image_buffer: scalar string Tensor representing the raw JPEG image buffer.\n    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]\n      where each coordinate is [0, 1) and the coordinates are arranged as\n      [ymin, xmin, ymax, xmax].\n    num_channels: Integer depth of the image buffer for decoding.\n\n  Returns:\n    3-D tensor with cropped image.\n\n  '
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(tf.image.extract_jpeg_shape(image_buffer), bounding_boxes=bbox, min_object_covered=0.1, aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1.0], max_attempts=100, use_image_if_no_bounding_boxes=True)
    (bbox_begin, bbox_size, _) = sample_distorted_bounding_box
    (offset_y, offset_x, _) = tf.unstack(bbox_begin)
    (target_height, target_width, _) = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    cropped = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=num_channels)
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped

def _central_crop(image, crop_height, crop_width):
    if False:
        print('Hello World!')
    'Performs central crops of the given image list.\n\n  Args:\n    image: a 3-D image tensor\n    crop_height: the height of the image following the crop.\n    crop_width: the width of the image following the crop.\n\n  Returns:\n    3-D tensor with cropped image.\n  '
    shape = tf.shape(input=image)
    (height, width) = (shape[0], shape[1])
    amount_to_be_cropped_h = height - crop_height
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = width - crop_width
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

def _mean_image_subtraction(image, means, num_channels):
    if False:
        return 10
    "Subtracts the given means from each image channel.\n\n  For example:\n    means = [123.68, 116.779, 103.939]\n    image = _mean_image_subtraction(image, means)\n\n  Note that the rank of `image` must be known.\n\n  Args:\n    image: a tensor of size [height, width, C].\n    means: a C-vector of values to subtract from each channel.\n    num_channels: number of color channels in the image that will be distorted.\n\n  Returns:\n    the centered image.\n\n  Raises:\n    ValueError: If the rank of `image` is unknown, if `image` has a rank other\n      than three or if the number of channels in `image` doesn't match the\n      number of values in `means`.\n  "
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    means = tf.broadcast_to(means, tf.shape(image))
    return image - means

def _smallest_size_at_least(height, width, resize_min):
    if False:
        while True:
            i = 10
    'Computes new shape with the smallest side equal to `smallest_side`.\n\n  Computes new shape with the smallest side equal to `smallest_side` while\n  preserving the original aspect ratio.\n\n  Args:\n    height: an int32 scalar tensor indicating the current height.\n    width: an int32 scalar tensor indicating the current width.\n    resize_min: A python integer or scalar `Tensor` indicating the size of\n      the smallest side after resize.\n\n  Returns:\n    new_height: an int32 scalar tensor indicating the new height.\n    new_width: an int32 scalar tensor indicating the new width.\n  '
    resize_min = tf.cast(resize_min, tf.float32)
    (height, width) = (tf.cast(height, tf.float32), tf.cast(width, tf.float32))
    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)
    return (new_height, new_width)

def _aspect_preserving_resize(image, resize_min):
    if False:
        while True:
            i = 10
    'Resize images preserving the original aspect ratio.\n\n  Args:\n    image: A 3-D image `Tensor`.\n    resize_min: A python integer or scalar `Tensor` indicating the size of\n      the smallest side after resize.\n\n  Returns:\n    resized_image: A 3-D tensor containing the resized image.\n  '
    shape = tf.shape(input=image)
    (height, width) = (shape[0], shape[1])
    (new_height, new_width) = _smallest_size_at_least(height, width, resize_min)
    return _resize_image(image, new_height, new_width)

def _resize_image(image, height, width):
    if False:
        i = 10
        return i + 15
    'Simple wrapper around tf.resize_images.\n\n  This is primarily to make sure we use the same `ResizeMethod` and other\n  details each time.\n\n  Args:\n    image: A 3-D image `Tensor`.\n    height: The target height for the resized image.\n    width: The target width for the resized image.\n\n  Returns:\n    resized_image: A 3-D tensor containing the resized image. The first two\n      dimensions have the shape [height, width].\n  '
    return tf.compat.v1.image.resize(image, [height, width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

def preprocess_image(image_buffer, bbox, output_height, output_width, num_channels, is_training=False):
    if False:
        return 10
    "Preprocesses the given image.\n\n  Preprocessing includes decoding, cropping, and resizing for both training\n  and eval images. Training preprocessing, however, introduces some random\n  distortion of the image to improve accuracy.\n\n  Args:\n    image_buffer: scalar string Tensor representing the raw JPEG image buffer.\n    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]\n      where each coordinate is [0, 1) and the coordinates are arranged as\n      [ymin, xmin, ymax, xmax].\n    output_height: The height of the image after preprocessing.\n    output_width: The width of the image after preprocessing.\n    num_channels: Integer depth of the image buffer for decoding.\n    is_training: `True` if we're preprocessing the image for training and\n      `False` otherwise.\n\n  Returns:\n    A preprocessed image.\n  "
    if is_training:
        image = _decode_crop_and_flip(image_buffer, bbox, num_channels)
        image = _resize_image(image, output_height, output_width)
    else:
        image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
        image = _aspect_preserving_resize(image, _RESIZE_MIN)
        image = _central_crop(image, output_height, output_width)
    image.set_shape([output_height, output_width, num_channels])
    return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)