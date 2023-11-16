"""Provides utilities to preprocess images for the Inception networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def apply_with_random_selector(x, func, num_cases):
    if False:
        for i in range(10):
            print('nop')
    'Computes func(x, sel), with sel sampled from [0...num_cases-1].\n\n  Args:\n    x: input Tensor.\n    func: Python function to apply.\n    num_cases: Python int32, number of cases to sample sel from.\n\n  Returns:\n    The result of func(x, sel), where func receives the value of the\n    selector as a python integer, but sel is sampled dynamically.\n  '
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    return control_flow_ops.merge([func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case) for case in range(num_cases)])[0]

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    if False:
        i = 10
        return i + 15
    'Distort the color of a Tensor image.\n\n  Each color distortion is non-commutative and thus ordering of the color ops\n  matters. Ideally we would randomly permute the ordering of the color ops.\n  Rather then adding that level of complication, we select a distinct ordering\n  of color ops for each preprocessing thread.\n\n  Args:\n    image: 3-D Tensor containing single image in [0, 1].\n    color_ordering: Python int, a type of distortion (valid values: 0-3).\n    fast_mode: Avoids slower ops (random_hue and random_contrast)\n    scope: Optional scope for name_scope.\n  Returns:\n    3-D Tensor color-distorted image on range [0, 1]\n  Raises:\n    ValueError: if color_ordering not in [0, 3]\n  '
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        elif color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        else:
            raise ValueError('color_ordering must be in [0, 3]')
        return tf.clip_by_value(image, 0.0, 1.0)

def distorted_bounding_box_crop(image, bbox, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 1.0), max_attempts=100, scope=None):
    if False:
        i = 10
        return i + 15
    'Generates cropped_image using a one of the bboxes randomly distorted.\n\n  See `tf.image.sample_distorted_bounding_box` for more documentation.\n\n  Args:\n    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).\n    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]\n      where each coordinate is [0, 1) and the coordinates are arranged\n      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole\n      image.\n    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped\n      area of the image must contain at least this fraction of any bounding box\n      supplied.\n    aspect_ratio_range: An optional list of `floats`. The cropped area of the\n      image must have an aspect ratio = width / height within this range.\n    area_range: An optional list of `floats`. The cropped area of the image\n      must contain a fraction of the supplied image within in this range.\n    max_attempts: An optional `int`. Number of attempts at generating a cropped\n      region of the image of the specified constraints. After `max_attempts`\n      failures, return the entire image.\n    scope: Optional scope for name_scope.\n  Returns:\n    A tuple, a 3-D Tensor cropped_image and the distorted bbox\n  '
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox, min_object_covered=min_object_covered, aspect_ratio_range=aspect_ratio_range, area_range=area_range, max_attempts=max_attempts, use_image_if_no_bounding_boxes=True)
        (bbox_begin, bbox_size, distort_bbox) = sample_distorted_bounding_box
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return (cropped_image, distort_bbox)

def preprocess_for_train(image, height, width, bbox, fast_mode=True, scope=None, add_image_summaries=True):
    if False:
        for i in range(10):
            print('nop')
    'Distort one image for training a network.\n\n  Distorting images provides a useful technique for augmenting the data\n  set during training in order to make the network invariant to aspects\n  of the image that do not effect the label.\n\n  Additionally it would create image_summaries to display the different\n  transformations applied to the image.\n\n  Args:\n    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be\n      [0, 1], otherwise it would converted to tf.float32 assuming that the range\n      is [0, MAX], where MAX is largest positive representable number for\n      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).\n    height: integer\n    width: integer\n    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]\n      where each coordinate is [0, 1) and the coordinates are arranged\n      as [ymin, xmin, ymax, xmax].\n    fast_mode: Optional boolean, if True avoids slower transformations (i.e.\n      bi-cubic resizing, random_hue or random_contrast).\n    scope: Optional scope for name_scope.\n    add_image_summaries: Enable image summaries.\n  Returns:\n    3-D float Tensor of distorted image used for training with range [-1, 1].\n  '
    with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
        if add_image_summaries:
            tf.summary.image('image_with_bounding_boxes', image_with_box)
        (distorted_image, distorted_bbox) = distorted_bounding_box_crop(image, bbox)
        distorted_image.set_shape([None, None, 3])
        image_with_distorted_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), distorted_bbox)
        if add_image_summaries:
            tf.summary.image('images_with_distorted_bounding_box', image_with_distorted_box)
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(distorted_image, lambda x, method: tf.image.resize_images(x, [height, width], method), num_cases=num_resize_cases)
        if add_image_summaries:
            tf.summary.image('cropped_resized_image', tf.expand_dims(distorted_image, 0))
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        num_distort_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(distorted_image, lambda x, ordering: distort_color(x, ordering, fast_mode), num_cases=num_distort_cases)
        if add_image_summaries:
            tf.summary.image('final_distorted_image', tf.expand_dims(distorted_image, 0))
        distorted_image = tf.subtract(distorted_image, 0.5)
        distorted_image = tf.multiply(distorted_image, 2.0)
        return distorted_image

def preprocess_for_eval(image, height, width, central_fraction=0.875, scope=None):
    if False:
        while True:
            i = 10
    'Prepare one image for evaluation.\n\n  If height and width are specified it would output an image with that size by\n  applying resize_bilinear.\n\n  If central_fraction is specified it would crop the central fraction of the\n  input image.\n\n  Args:\n    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be\n      [0, 1], otherwise it would converted to tf.float32 assuming that the range\n      is [0, MAX], where MAX is largest positive representable number for\n      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).\n    height: integer\n    width: integer\n    central_fraction: Optional Float, fraction of the image to crop.\n    scope: Optional scope for name_scope.\n  Returns:\n    3-D float Tensor of prepared image.\n  '
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)
        if height and width:
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

def preprocess_image(image, height, width, is_training=False, bbox=None, fast_mode=True, add_image_summaries=True):
    if False:
        i = 10
        return i + 15
    'Pre-process one image for training or evaluation.\n\n  Args:\n    image: 3-D Tensor [height, width, channels] with the image. If dtype is\n      tf.float32 then the range should be [0, 1], otherwise it would converted\n      to tf.float32 assuming that the range is [0, MAX], where MAX is largest\n      positive representable number for int(8/16/32) data type (see\n      `tf.image.convert_image_dtype` for details).\n    height: integer, image expected height.\n    width: integer, image expected width.\n    is_training: Boolean. If true it would transform an image for train,\n      otherwise it would transform it for evaluation.\n    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]\n      where each coordinate is [0, 1) and the coordinates are arranged as\n      [ymin, xmin, ymax, xmax].\n    fast_mode: Optional boolean, if True avoids slower transformations.\n    add_image_summaries: Enable image summaries.\n\n  Returns:\n    3-D float Tensor containing an appropriately scaled image\n\n  Raises:\n    ValueError: if user does not provide bounding box\n  '
    if is_training:
        return preprocess_for_train(image, height, width, bbox, fast_mode, add_image_summaries=add_image_summaries)
    else:
        return preprocess_for_eval(image, height, width)