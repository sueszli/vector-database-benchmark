"""Imagenet input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import flags
import tensorflow as tf
FLAGS = flags.FLAGS
flags.DEFINE_string('imagenet_data_dir', None, 'Directory with Imagenet dataset in TFRecord format.')

def _decode_and_random_crop(image_buffer, bbox, image_size):
    if False:
        print('Hello World!')
    'Randomly crops image and then scales to target size.'
    with tf.name_scope('distorted_bounding_box_crop', values=[image_buffer, bbox]):
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(tf.image.extract_jpeg_shape(image_buffer), bounding_boxes=bbox, min_object_covered=0.1, aspect_ratio_range=[0.75, 1.33], area_range=[0.08, 1.0], max_attempts=10, use_image_if_no_bounding_boxes=True)
        (bbox_begin, bbox_size, _) = sample_distorted_bounding_box
        (offset_y, offset_x, _) = tf.unstack(bbox_begin)
        (target_height, target_width, _) = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
        image = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
        return image

def _decode_and_center_crop(image_buffer, image_size):
    if False:
        print('Hello World!')
    'Crops to center of image with padding then scales to target size.'
    shape = tf.image.extract_jpeg_shape(image_buffer)
    image_height = shape[0]
    image_width = shape[1]
    padded_center_crop_size = tf.cast(0.875 * tf.cast(tf.minimum(image_height, image_width), tf.float32), tf.int32)
    offset_height = (image_height - padded_center_crop_size + 1) // 2
    offset_width = (image_width - padded_center_crop_size + 1) // 2
    crop_window = tf.stack([offset_height, offset_width, padded_center_crop_size, padded_center_crop_size])
    image = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
    return image

def _normalize(image):
    if False:
        i = 10
        return i + 15
    'Rescale image to [-1, 1] range.'
    return tf.multiply(tf.subtract(image, 0.5), 2.0)

def image_preprocessing(image_buffer, bbox, image_size, is_training):
    if False:
        return 10
    'Does image decoding and preprocessing.\n\n  Args:\n    image_buffer: string tensor with encoded image.\n    bbox: bounding box of the object at the image.\n    image_size: image size.\n    is_training: whether to do training or eval preprocessing.\n\n  Returns:\n    Tensor with the image.\n  '
    if is_training:
        image = _decode_and_random_crop(image_buffer, bbox, image_size)
        image = _normalize(image)
        image = tf.image.random_flip_left_right(image)
    else:
        image = _decode_and_center_crop(image_buffer, image_size)
        image = _normalize(image)
    image = tf.reshape(image, [image_size, image_size, 3])
    return image

def imagenet_parser(value, image_size, is_training):
    if False:
        return 10
    'Parse an ImageNet record from a serialized string Tensor.\n\n  Args:\n    value: encoded example.\n    image_size: size of the output image.\n    is_training: if True then do training preprocessing,\n      otherwise do eval preprocessing.\n\n  Returns:\n    image: tensor with the image.\n    label: true label of the image.\n  '
    keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, ''), 'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'), 'image/class/label': tf.FixedLenFeature([], tf.int64, -1), 'image/class/text': tf.FixedLenFeature([], tf.string, ''), 'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32), 'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32), 'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32), 'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32), 'image/object/class/label': tf.VarLenFeature(dtype=tf.int64)}
    parsed = tf.parse_single_example(value, keys_to_features)
    image_buffer = tf.reshape(parsed['image/encoded'], shape=[])
    xmin = tf.expand_dims(parsed['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(parsed['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(parsed['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(parsed['image/object/bbox/ymax'].values, 0)
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])
    image = image_preprocessing(image_buffer=image_buffer, bbox=bbox, image_size=image_size, is_training=is_training)
    label = tf.cast(tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32)
    return (image, label)

def imagenet_input(split, batch_size, image_size, is_training):
    if False:
        print('Hello World!')
    'Returns ImageNet dataset.\n\n  Args:\n    split: name of the split, "train" or "validation".\n    batch_size: size of the minibatch.\n    image_size: size of the one side of the image. Output images will be\n      resized to square shape image_size*image_size.\n    is_training: if True then training preprocessing is done, otherwise eval\n      preprocessing is done.\n\n  Raises:\n    ValueError: if name of the split is incorrect.\n\n  Returns:\n    Instance of tf.data.Dataset with the dataset.\n  '
    if split.lower().startswith('train'):
        file_pattern = os.path.join(FLAGS.imagenet_data_dir, 'train-*')
    elif split.lower().startswith('validation'):
        file_pattern = os.path.join(FLAGS.imagenet_data_dir, 'validation-*')
    else:
        raise ValueError('Invalid split: %s' % split)
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
    if is_training:
        dataset = dataset.repeat()

    def fetch_dataset(filename):
        if False:
            while True:
                i = 10
        return tf.data.TFRecordDataset(filename, buffer_size=8 * 1024 * 1024)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(fetch_dataset, cycle_length=4, sloppy=True))
    dataset = dataset.shuffle(1024)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(lambda value: imagenet_parser(value, image_size, is_training), batch_size=batch_size, num_parallel_batches=4, drop_remainder=True))

    def set_shapes(images, labels):
        if False:
            return 10
        'Statically set the batch_size dimension.'
        images.set_shape(images.get_shape().merge_with(tf.TensorShape([batch_size, None, None, None])))
        labels.set_shape(labels.get_shape().merge_with(tf.TensorShape([batch_size])))
        return (images, labels)
    dataset = dataset.map(set_shapes)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def num_examples_per_epoch(split):
    if False:
        print('Hello World!')
    'Returns the number of examples in the data set.\n\n  Args:\n    split: name of the split, "train" or "validation".\n\n  Raises:\n    ValueError: if split name is incorrect.\n\n  Returns:\n    Number of example in the split.\n  '
    if split.lower().startswith('train'):
        return 1281167
    elif split.lower().startswith('validation'):
        return 50000
    else:
        raise ValueError('Invalid split: %s' % split)