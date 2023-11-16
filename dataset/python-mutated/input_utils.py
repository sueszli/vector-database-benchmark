"""Utility functions for input processing."""
import math
import tensorflow.compat.v2 as tf
from official.vision.detection.utils import box_utils
from official.vision.detection.utils.object_detection import preprocessor

def pad_to_fixed_size(input_tensor, size, constant_values=0):
    if False:
        i = 10
        return i + 15
    'Pads data to a fixed length at the first dimension.\n\n  Args:\n    input_tensor: `Tensor` with any dimension.\n    size: `int` number for the first dimension of output Tensor.\n    constant_values: `int` value assigned to the paddings.\n\n  Returns:\n    `Tensor` with the first dimension padded to `size`.\n  '
    input_shape = input_tensor.get_shape().as_list()
    padding_shape = []
    padding_length = size - tf.shape(input=input_tensor)[0]
    assert_length = tf.Assert(tf.greater_equal(padding_length, 0), [padding_length])
    with tf.control_dependencies([assert_length]):
        padding_shape.append(padding_length)
    for i in range(1, len(input_shape)):
        padding_shape.append(tf.shape(input=input_tensor)[i])
    paddings = tf.cast(constant_values * tf.ones(padding_shape), input_tensor.dtype)
    padded_tensor = tf.concat([input_tensor, paddings], axis=0)
    output_shape = input_shape
    output_shape[0] = size
    padded_tensor.set_shape(output_shape)
    return padded_tensor

def normalize_image(image, offset=(0.485, 0.456, 0.406), scale=(0.229, 0.224, 0.225)):
    if False:
        for i in range(10):
            print('nop')
    'Normalizes the image to zero mean and unit variance.'
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    image -= offset
    scale = tf.constant(scale)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    image /= scale
    return image

def compute_padded_size(desired_size, stride):
    if False:
        print('Hello World!')
    'Compute the padded size given the desired size and the stride.\n\n  The padded size will be the smallest rectangle, such that each dimension is\n  the smallest multiple of the stride which is larger than the desired\n  dimension. For example, if desired_size = (100, 200) and stride = 32,\n  the output padded_size = (128, 224).\n\n  Args:\n    desired_size: a `Tensor` or `int` list/tuple of two elements representing\n      [height, width] of the target output image size.\n    stride: an integer, the stride of the backbone network.\n\n  Returns:\n    padded_size: a `Tensor` or `int` list/tuple of two elements representing\n      [height, width] of the padded output image size.\n  '
    if isinstance(desired_size, list) or isinstance(desired_size, tuple):
        padded_size = [int(math.ceil(d * 1.0 / stride) * stride) for d in desired_size]
    else:
        padded_size = tf.cast(tf.math.ceil(tf.cast(desired_size, dtype=tf.float32) / stride) * stride, tf.int32)
    return padded_size

def resize_and_crop_image(image, desired_size, padded_size, aug_scale_min=1.0, aug_scale_max=1.0, seed=1, method=tf.image.ResizeMethod.BILINEAR):
    if False:
        i = 10
        return i + 15
    'Resizes the input image to output size.\n\n  Resize and pad images given the desired output size of the image and\n  stride size.\n\n  Here are the preprocessing steps.\n  1. For a given image, keep its aspect ratio and rescale the image to make it\n     the largest rectangle to be bounded by the rectangle specified by the\n     `desired_size`.\n  2. Pad the rescaled image to the padded_size.\n\n  Args:\n    image: a `Tensor` of shape [height, width, 3] representing an image.\n    desired_size: a `Tensor` or `int` list/tuple of two elements representing\n      [height, width] of the desired actual output image size.\n    padded_size: a `Tensor` or `int` list/tuple of two elements representing\n      [height, width] of the padded output image size. Padding will be applied\n      after scaling the image to the desired_size.\n    aug_scale_min: a `float` with range between [0, 1.0] representing minimum\n      random scale applied to desired_size for training scale jittering.\n    aug_scale_max: a `float` with range between [1.0, inf] representing maximum\n      random scale applied to desired_size for training scale jittering.\n    seed: seed for random scale jittering.\n    method: function to resize input image to scaled image.\n\n  Returns:\n    output_image: `Tensor` of shape [height, width, 3] where [height, width]\n      equals to `output_size`.\n    image_info: a 2D `Tensor` that encodes the information of the image and the\n      applied preprocessing. It is in the format of\n      [[original_height, original_width], [scaled_height, scaled_width],\n       [y_scale, x_scale], [y_offset, x_offset]], where [scaled_height,\n      scaled_width] is the actual scaled image size, and [y_scale, x_scale] is\n      the scaling factory, which is the ratio of\n      scaled dimension / original dimension.\n  '
    with tf.name_scope('resize_and_crop_image'):
        image_size = tf.cast(tf.shape(input=image)[0:2], tf.float32)
        random_jittering = aug_scale_min != 1.0 or aug_scale_max != 1.0
        if random_jittering:
            random_scale = tf.random.uniform([], aug_scale_min, aug_scale_max, seed=seed)
            scaled_size = tf.round(random_scale * desired_size)
        else:
            scaled_size = desired_size
        scale = tf.minimum(scaled_size[0] / image_size[0], scaled_size[1] / image_size[1])
        scaled_size = tf.round(image_size * scale)
        image_scale = scaled_size / image_size
        if random_jittering:
            max_offset = scaled_size - desired_size
            max_offset = tf.where(tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset)
            offset = max_offset * tf.random.uniform([2], 0, 1, seed=seed)
            offset = tf.cast(offset, tf.int32)
        else:
            offset = tf.zeros((2,), tf.int32)
        scaled_image = tf.image.resize(image, tf.cast(scaled_size, tf.int32), method=method)
        if random_jittering:
            scaled_image = scaled_image[offset[0]:offset[0] + desired_size[0], offset[1]:offset[1] + desired_size[1], :]
        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, padded_size[0], padded_size[1])
        image_info = tf.stack([image_size, scaled_size, image_scale, tf.cast(offset, tf.float32)])
        return (output_image, image_info)

def resize_and_crop_image_v2(image, short_side, long_side, padded_size, aug_scale_min=1.0, aug_scale_max=1.0, seed=1, method=tf.image.ResizeMethod.BILINEAR):
    if False:
        i = 10
        return i + 15
    'Resizes the input image to output size (Faster R-CNN style).\n\n  Resize and pad images given the specified short / long side length and the\n  stride size.\n\n  Here are the preprocessing steps.\n  1. For a given image, keep its aspect ratio and first try to rescale the short\n     side of the original image to `short_side`.\n  2. If the scaled image after 1 has a long side that exceeds `long_side`, keep\n     the aspect ratio and rescal the long side of the image to `long_side`.\n  2. Pad the rescaled image to the padded_size.\n\n  Args:\n    image: a `Tensor` of shape [height, width, 3] representing an image.\n    short_side: a scalar `Tensor` or `int` representing the desired short side\n      to be rescaled to.\n    long_side: a scalar `Tensor` or `int` representing the desired long side to\n      be rescaled to.\n    padded_size: a `Tensor` or `int` list/tuple of two elements representing\n      [height, width] of the padded output image size. Padding will be applied\n      after scaling the image to the desired_size.\n    aug_scale_min: a `float` with range between [0, 1.0] representing minimum\n      random scale applied to desired_size for training scale jittering.\n    aug_scale_max: a `float` with range between [1.0, inf] representing maximum\n      random scale applied to desired_size for training scale jittering.\n    seed: seed for random scale jittering.\n    method: function to resize input image to scaled image.\n\n  Returns:\n    output_image: `Tensor` of shape [height, width, 3] where [height, width]\n      equals to `output_size`.\n    image_info: a 2D `Tensor` that encodes the information of the image and the\n      applied preprocessing. It is in the format of\n      [[original_height, original_width], [scaled_height, scaled_width],\n       [y_scale, x_scale], [y_offset, x_offset]], where [scaled_height,\n      scaled_width] is the actual scaled image size, and [y_scale, x_scale] is\n      the scaling factor, which is the ratio of\n      scaled dimension / original dimension.\n  '
    with tf.name_scope('resize_and_crop_image_v2'):
        image_size = tf.cast(tf.shape(image)[0:2], tf.float32)
        scale_using_short_side = short_side / tf.math.minimum(image_size[0], image_size[1])
        scale_using_long_side = long_side / tf.math.maximum(image_size[0], image_size[1])
        scaled_size = tf.math.round(image_size * scale_using_short_side)
        scaled_size = tf.where(tf.math.greater(tf.math.maximum(scaled_size[0], scaled_size[1]), long_side), tf.math.round(image_size * scale_using_long_side), scaled_size)
        desired_size = scaled_size
        random_jittering = aug_scale_min != 1.0 or aug_scale_max != 1.0
        if random_jittering:
            random_scale = tf.random.uniform([], aug_scale_min, aug_scale_max, seed=seed)
            scaled_size = tf.math.round(random_scale * scaled_size)
        image_scale = scaled_size / image_size
        if random_jittering:
            max_offset = scaled_size - desired_size
            max_offset = tf.where(tf.math.less(max_offset, 0), tf.zeros_like(max_offset), max_offset)
            offset = max_offset * tf.random.uniform([2], 0, 1, seed=seed)
            offset = tf.cast(offset, tf.int32)
        else:
            offset = tf.zeros((2,), tf.int32)
        scaled_image = tf.image.resize(image, tf.cast(scaled_size, tf.int32), method=method)
        if random_jittering:
            scaled_image = scaled_image[offset[0]:offset[0] + desired_size[0], offset[1]:offset[1] + desired_size[1], :]
        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, padded_size[0], padded_size[1])
        image_info = tf.stack([image_size, scaled_size, image_scale, tf.cast(offset, tf.float32)])
        return (output_image, image_info)

def resize_and_crop_boxes(boxes, image_scale, output_size, offset):
    if False:
        i = 10
        return i + 15
    'Resizes boxes to output size with scale and offset.\n\n  Args:\n    boxes: `Tensor` of shape [N, 4] representing ground truth boxes.\n    image_scale: 2D float `Tensor` representing scale factors that apply to\n      [height, width] of input image.\n    output_size: 2D `Tensor` or `int` representing [height, width] of target\n      output image size.\n    offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled\n      boxes.\n\n  Returns:\n    boxes: `Tensor` of shape [N, 4] representing the scaled boxes.\n  '
    boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
    boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
    boxes = box_utils.clip_boxes(boxes, output_size)
    return boxes

def resize_and_crop_masks(masks, image_scale, output_size, offset):
    if False:
        print('Hello World!')
    'Resizes boxes to output size with scale and offset.\n\n  Args:\n    masks: `Tensor` of shape [N, H, W, 1] representing ground truth masks.\n    image_scale: 2D float `Tensor` representing scale factors that apply to\n      [height, width] of input image.\n    output_size: 2D `Tensor` or `int` representing [height, width] of target\n      output image size.\n    offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled\n      boxes.\n\n  Returns:\n    masks: `Tensor` of shape [N, H, W, 1] representing the scaled masks.\n  '
    mask_size = tf.shape(input=masks)[1:3]
    scaled_size = tf.cast(image_scale * tf.cast(mask_size, image_scale.dtype), tf.int32)
    scaled_masks = tf.image.resize(masks, scaled_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    offset = tf.cast(offset, tf.int32)
    scaled_masks = scaled_masks[:, offset[0]:offset[0] + output_size[0], offset[1]:offset[1] + output_size[1], :]
    output_masks = tf.image.pad_to_bounding_box(scaled_masks, 0, 0, output_size[0], output_size[1])
    return output_masks

def random_horizontal_flip(image, boxes=None, masks=None):
    if False:
        for i in range(10):
            print('nop')
    'Randomly flips input image and bounding boxes.'
    return preprocessor.random_horizontal_flip(image, boxes, masks)