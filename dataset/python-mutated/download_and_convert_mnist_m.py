"""Downloads and converts MNIST-M data to TFRecords of TF-Example protos.

This module downloads the MNIST-M data, uncompresses it, reads the files
that make up the MNIST-M data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import sys
import numpy as np
from six.moves import urllib
import tensorflow as tf
from slim.datasets import dataset_utils
tf.app.flags.DEFINE_string('dataset_dir', None, 'The directory where the output TFRecords and temporary files are saved.')
FLAGS = tf.app.flags.FLAGS
_IMAGE_SIZE = 32
_NUM_CHANNELS = 3
_NUM_TRAIN_SAMPLES = 59001
_NUM_VALIDATION = 1000
_NUM_TEST_SAMPLES = 9001
_RANDOM_SEED = 0
_CLASS_NAMES = ['zero', 'one', 'two', 'three', 'four', 'five', 'size', 'seven', 'eight', 'nine']

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        if False:
            print('Hello World!')
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

    def read_image_dims(self, sess, image_data):
        if False:
            i = 10
            return i + 15
        image = self.decode_png(sess, image_data)
        return (image.shape[0], image.shape[1])

    def decode_png(self, sess, image_data):
        if False:
            i = 10
            return i + 15
        image = sess.run(self._decode_png, feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _convert_dataset(split_name, filenames, filename_to_class_id, dataset_dir):
    if False:
        print('Hello World!')
    "Converts the given filenames to a TFRecord dataset.\n\n  Args:\n    split_name: The name of the dataset, either 'train' or 'valid'.\n    filenames: A list of absolute paths to png images.\n    filename_to_class_id: A dictionary from filenames (strings) to class ids\n      (integers).\n    dataset_dir: The directory where the converted datasets are stored.\n  "
    print('Converting the {} split.'.format(split_name))
    if split_name in ['train', 'valid']:
        png_directory = os.path.join(dataset_dir, 'mnist_m', 'mnist_m_train')
    elif split_name == 'test':
        png_directory = os.path.join(dataset_dir, 'mnist_m', 'mnist_m_test')
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:
            output_filename = _get_output_filename(dataset_dir, split_name)
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for filename in filenames:
                    image_data = tf.gfile.FastGFile(os.path.join(png_directory, filename), 'r').read()
                    (height, width) = image_reader.read_image_dims(sess, image_data)
                    class_id = filename_to_class_id[filename]
                    example = dataset_utils.image_to_tfexample(image_data, 'png', height, width, class_id)
                    tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def _extract_labels(label_filename):
    if False:
        return 10
    'Extract the labels into a dict of filenames to int labels.\n\n  Args:\n    labels_filename: The filename of the MNIST-M labels.\n\n  Returns:\n    A dictionary of filenames to int labels.\n  '
    print('Extracting labels from: ', label_filename)
    label_file = tf.gfile.FastGFile(label_filename, 'r').readlines()
    label_lines = [line.rstrip('\n').split() for line in label_file]
    labels = {}
    for line in label_lines:
        assert len(line) == 2
        labels[line[0]] = int(line[1])
    return labels

def _get_output_filename(dataset_dir, split_name):
    if False:
        while True:
            i = 10
    'Creates the output filename.\n\n  Args:\n    dataset_dir: The directory where the temporary files are stored.\n    split_name: The name of the train/test split.\n\n  Returns:\n    An absolute file path.\n  '
    return '%s/mnist_m_%s.tfrecord' % (dataset_dir, split_name)

def _get_filenames(dataset_dir):
    if False:
        while True:
            i = 10
    'Returns a list of filenames and inferred class names.\n\n  Args:\n    dataset_dir: A directory containing a set PNG encoded MNIST-M images.\n\n  Returns:\n    A list of image file paths, relative to `dataset_dir`.\n  '
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        photo_filenames.append(filename)
    return photo_filenames

def run(dataset_dir):
    if False:
        return 10
    'Runs the download and conversion operation.\n\n  Args:\n    dataset_dir: The dataset directory where the dataset is stored.\n  '
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    train_filename = _get_output_filename(dataset_dir, 'train')
    testing_filename = _get_output_filename(dataset_dir, 'test')
    if tf.gfile.Exists(train_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    train_validation_filenames = _get_filenames(os.path.join(dataset_dir, 'mnist_m', 'mnist_m_train'))
    test_filenames = _get_filenames(os.path.join(dataset_dir, 'mnist_m', 'mnist_m_test'))
    random.seed(_RANDOM_SEED)
    random.shuffle(train_validation_filenames)
    train_filenames = train_validation_filenames[_NUM_VALIDATION:]
    validation_filenames = train_validation_filenames[:_NUM_VALIDATION]
    train_validation_filenames_to_class_ids = _extract_labels(os.path.join(dataset_dir, 'mnist_m', 'mnist_m_train_labels.txt'))
    test_filenames_to_class_ids = _extract_labels(os.path.join(dataset_dir, 'mnist_m', 'mnist_m_test_labels.txt'))
    _convert_dataset('train', train_filenames, train_validation_filenames_to_class_ids, dataset_dir)
    _convert_dataset('valid', validation_filenames, train_validation_filenames_to_class_ids, dataset_dir)
    _convert_dataset('test', test_filenames, test_filenames_to_class_ids, dataset_dir)
    labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MNIST-M dataset!')

def main(_):
    if False:
        for i in range(10):
            print('nop')
    assert FLAGS.dataset_dir
    run(FLAGS.dataset_dir)
if __name__ == '__main__':
    tf.app.run()