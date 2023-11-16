"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import tarfile
import zipfile
from six.moves import urllib
import tensorflow as tf
LABELS_FILENAME = 'labels.txt'

def int64_feature(values):
    if False:
        return 10
    'Returns a TF-Feature of int64s.\n\n  Args:\n    values: A scalar or list of values.\n\n  Returns:\n    A TF-Feature.\n  '
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_list_feature(values):
    if False:
        print('Hello World!')
    'Returns a TF-Feature of list of bytes.\n\n  Args:\n    values: A string or list of strings.\n\n  Returns:\n    A TF-Feature.\n  '
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def float_list_feature(values):
    if False:
        print('Hello World!')
    'Returns a TF-Feature of list of floats.\n\n  Args:\n    values: A float or list of floats.\n\n  Returns:\n    A TF-Feature.\n  '
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def bytes_feature(values):
    if False:
        while True:
            i = 10
    'Returns a TF-Feature of bytes.\n\n  Args:\n    values: A string.\n\n  Returns:\n    A TF-Feature.\n  '
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def float_feature(values):
    if False:
        print('Hello World!')
    'Returns a TF-Feature of floats.\n\n  Args:\n    values: A scalar of list of values.\n\n  Returns:\n    A TF-Feature.\n  '
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def image_to_tfexample(image_data, image_format, height, width, class_id):
    if False:
        print('Hello World!')
    return tf.train.Example(features=tf.train.Features(feature={'image/encoded': bytes_feature(image_data), 'image/format': bytes_feature(image_format), 'image/class/label': int64_feature(class_id), 'image/height': int64_feature(height), 'image/width': int64_feature(width)}))

def download_url(url, dataset_dir):
    if False:
        i = 10
        return i + 15
    'Downloads the tarball or zip file from url into filepath.\n\n  Args:\n    url: The URL of a tarball or zip file.\n    dataset_dir: The directory where the temporary files are stored.\n\n  Returns:\n    filepath: path where the file is downloaded.\n  '
    filename = url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
        if False:
            print('Hello World!')
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    (filepath, _) = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath

def download_and_uncompress_tarball(tarball_url, dataset_dir):
    if False:
        for i in range(10):
            print('nop')
    'Downloads the `tarball_url` and uncompresses it locally.\n\n  Args:\n    tarball_url: The URL of a tarball file.\n    dataset_dir: The directory where the temporary files are stored.\n  '
    filepath = download_url(tarball_url, dataset_dir)
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)

def download_and_uncompress_zipfile(zip_url, dataset_dir):
    if False:
        print('Hello World!')
    'Downloads the `zip_url` and uncompresses it locally.\n\n  Args:\n    zip_url: The URL of a zip file.\n    dataset_dir: The directory where the temporary files are stored.\n  '
    filename = zip_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    if tf.gfile.Exists(filepath):
        print('File {filename} has been already downloaded at {filepath}. Unzipping it....'.format(filename=filename, filepath=filepath))
    else:
        filepath = download_url(zip_url, dataset_dir)
    with zipfile.ZipFile(filepath, 'r') as zip_file:
        for member in zip_file.namelist():
            memberpath = os.path.join(dataset_dir, member)
            if not (os.path.exists(memberpath) or os.path.isfile(memberpath)):
                zip_file.extract(member, dataset_dir)

def write_label_file(labels_to_class_names, dataset_dir, filename=LABELS_FILENAME):
    if False:
        i = 10
        return i + 15
    'Writes a file with the list of class names.\n\n  Args:\n    labels_to_class_names: A map of (integer) labels to class names.\n    dataset_dir: The directory in which the labels file should be written.\n    filename: The filename where the class names are written.\n  '
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))

def has_labels(dataset_dir, filename=LABELS_FILENAME):
    if False:
        while True:
            i = 10
    'Specifies whether or not the dataset directory contains a label map file.\n\n  Args:\n    dataset_dir: The directory in which the labels file is found.\n    filename: The filename where the class names are written.\n\n  Returns:\n    `True` if the labels file exists and `False` otherwise.\n  '
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))

def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    if False:
        for i in range(10):
            print('nop')
    'Reads the labels file and returns a mapping from ID to class name.\n\n  Args:\n    dataset_dir: The directory in which the labels file is found.\n    filename: The filename where the class names are written.\n\n  Returns:\n    A map from a label (integer) to class name.\n  '
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)
    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    return labels_to_class_names

def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    if False:
        print('Hello World!')
    'Opens all TFRecord shards for writing and adds them to an exit stack.\n\n  Args:\n    exit_stack: A context2.ExitStack used to automatically closed the TFRecords\n      opened in this function.\n    base_path: The base path for all shards\n    num_shards: The number of shards\n\n  Returns:\n    The list of opened TFRecords. Position k in the list corresponds to shard k.\n  '
    tf_record_output_filenames = ['{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards) for idx in range(num_shards)]
    tfrecords = [exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name)) for file_name in tf_record_output_filenames]
    return tfrecords