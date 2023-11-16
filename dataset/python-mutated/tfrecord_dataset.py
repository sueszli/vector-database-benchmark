import os
from bigdl.orca.data.image.imagenet_dataset import *
from bigdl.orca.data.image.parquet_dataset import _check_arguments
from bigdl.dllib.utils.log4Error import invalidInputError
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from tensorflow.data import Dataset

def write_imagenet(imagenet_path: str, output_path: str, **kwargs) -> None:
    if False:
        print('Hello World!')
    "\n    Write ImageNet data to TFRecords file format. The train and validation data will be\n    converted into 1024 and 128 TFRecord files, respectively. Each train TFRecord file\n    contains ~1250 records. Each validation TFRecord file contains ~390 records.\n\n    Each record within the TFRecord file is a serialized Example proto. The Example proto\n    contains the following fields:\n\n    image/height: integer, image height in pixels\n    image/width: integer, image width in pixels\n    image/colorspace: string, specifying the colorspace, always 'RGB'\n    image/channels: integer, specifying the number of channels, always 3\n    image/class/label: integer, identifier for the ground truth for the network\n    image/class/synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'\n    image/format: string, specifying the format, always'JPEG'\n    image/filename: string, path to an image file, e.g., '/path/to/example.JPG'\n    image/encoded: string containing JPEG encoded image in RGB colorspace\n\n    Args:\n    imagenet_path: ImageNet raw data path. Download raw ImageNet data from\n                   http://image-net.org/download-images\n                   e.g, if you use ImageNet 2012, please extract ILSVRC2012_img_train.tar and\n                   ILSVRC2012_img_val.tar. Download the validation image labels file\n                   https://github.com/tensorflow/models/blob/master/research/slim/datasets/\n                   imagenet_2012_validation_synset_labels.txt and rename as synset_labels.txt\n                   provide imagenet path in this format:\n                   - Training images: train/n03062245/n03062245_4620.JPEG\n                   - Validation Images: validation/ILSVRC2012_val_00000001.JPEG\n                   - Validation Labels: synset_labels.txt\n    output_path: Output data directory\n\n    "
    if not imagenet_path:
        invalidInputError(False, 'ImageNet data path should not be empty. Please download from http://image-net.org/download-images and extract .tar and provide raw data directory path')
    convert_imagenet_to_tf_records(imagenet_path, output_path, **kwargs)

def read_imagenet(path: str, is_training: bool) -> 'Dataset':
    if False:
        i = 10
        return i + 15
    '\n    Convert ImageNet TFRecords files to tf.data.Dataset\n\n    Args:\n    data_dir: ImageNet TFRecords data path. It supports local path or hdfs path. If you use\n              hdfs path, please make sure set environment variables LD_LIBRARY_PATH within PATH.\n            - Training images: train/train-00000-of-01024\n            - Validation Images: validation/validation-00000-of-00128\n    is_training: True or False. train dataset or val dataset\n\n    '
    import tensorflow as tf
    filenames = get_filenames(is_training, path)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def get_filenames(is_training: bool, data_dir: str) -> List[str]:
    if False:
        while True:
            i = 10
    'Return filenames for dataset.'
    _NUM_IMAGENET_TRAIN_FILES = 1024
    _NUM_IMAGENET_VAL_FILES = 128
    if is_training:
        return [os.path.join(data_dir, 'train-%05d-of-01024' % i) for i in range(_NUM_IMAGENET_TRAIN_FILES)]
    else:
        return [os.path.join(data_dir, 'validation-%05d-of-00128' % i) for i in range(_NUM_IMAGENET_VAL_FILES)]

def write_tfrecord(format: str, output_path: str, *args, **kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert input dataset to TFRecords\n\n    Args:\n    format: String. Support "imagenet" format.\n    output_path: String. output path.\n\n    '
    supported_format = {'imagenet'}
    if format not in supported_format:
        invalidInputError(False, format + " is not supported, should be 'imagenet'. ")
    format_to_function = {'imagenet': (write_imagenet, ['imagenet_path'])}
    (func, required_args) = format_to_function[format]
    _check_arguments(format, kwargs, required_args)
    kwargs['output_path'] = output_path
    func(*args, **kwargs)

def read_tfrecord(format: str, path: str, *args, **kwargs) -> 'Dataset':
    if False:
        for i in range(10):
            print('nop')
    '\n    Read TFRecords files\n\n    Args:\n    format: String. Support "imagenet" format.\n    path: String. TFRecords files path.\n\n    '
    supported_format = {'imagenet'}
    if format not in supported_format:
        invalidInputError(False, format + " is not supported, should be 'imagenet'. ")
    format_to_function = {'imagenet': (read_imagenet, ['is_training'])}
    (func, required_args) = format_to_function[format]
    _check_arguments(format, kwargs, required_args)
    kwargs['path'] = path
    return func(*args, **kwargs)