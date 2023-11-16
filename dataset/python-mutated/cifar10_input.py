"""Routine for decoding the CIFAR-10 binary file format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow_datasets as tfds
IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def _get_images_labels(batch_size, split, distords=False):
    if False:
        while True:
            i = 10
    'Returns Dataset for given split.'
    dataset = tfds.load(name='cifar10', split=split)
    scope = 'data_augmentation' if distords else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(distords), num_parallel_calls=10)
    dataset = dataset.prefetch(-1)
    dataset = dataset.repeat().batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images_labels = iterator.get_next()
    (images, labels) = (images_labels['input'], images_labels['target'])
    tf.summary.image('images', images)
    return (images, labels)

class DataPreprocessor(object):
    """Applies transformations to dataset record."""

    def __init__(self, distords):
        if False:
            for i in range(10):
                print('nop')
        self._distords = distords

    def __call__(self, record):
        if False:
            i = 10
            return i + 15
        'Process img for training or eval.'
        img = record['image']
        img = tf.cast(img, tf.float32)
        if self._distords:
            img = tf.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=63)
            img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        else:
            img = tf.image.resize_image_with_crop_or_pad(img, IMAGE_SIZE, IMAGE_SIZE)
        img = tf.image.per_image_standardization(img)
        return dict(input=img, target=record['label'])

def distorted_inputs(batch_size):
    if False:
        i = 10
        return i + 15
    'Construct distorted input for CIFAR training using the Reader ops.\n\n  Args:\n    batch_size: Number of images per batch.\n\n  Returns:\n    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.\n    labels: Labels. 1D tensor of [batch_size] size.\n  '
    return _get_images_labels(batch_size, tfds.Split.TRAIN, distords=True)

def inputs(eval_data, batch_size):
    if False:
        i = 10
        return i + 15
    'Construct input for CIFAR evaluation using the Reader ops.\n\n  Args:\n    eval_data: bool, indicating if one should use the train or eval data set.\n    batch_size: Number of images per batch.\n\n  Returns:\n    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.\n    labels: Labels. 1D tensor of [batch_size] size.\n  '
    split = tfds.Split.TEST if eval_data == 'test' else tfds.Split.TRAIN
    return _get_images_labels(batch_size, split)