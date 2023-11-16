"""Converts Cityscapes data to TFRecord file format with Example protos.

The Cityscapes dataset is expected to have the following directory structure:

  + cityscapes
     - build_cityscapes_data.py (current working directiory).
     - build_data.py
     + cityscapesscripts
       + annotation
       + evaluation
       + helpers
       + preparation
       + viewer
     + gtFine
       + train
       + val
       + test
     + leftImg8bit
       + train
       + val
       + test
     + tfrecord

This script converts data into sharded data files and save at tfrecord folder.

Note that before running this script, the users should (1) register the
Cityscapes dataset website at https://www.cityscapes-dataset.com to
download the dataset, and (2) run the script provided by Cityscapes
`preparation/createTrainIdLabelImgs.py` to generate the training groundtruth.

Also note that the tensorflow model will be trained with `TrainId' instead
of `EvalId' used on the evaluation server. Thus, the users need to convert
the predicted labels to `EvalId` for evaluation on the server. See the
vis.py for more details.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import math
import os.path
import re
import sys
import build_data
from six.moves import range
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cityscapes_root', './cityscapes', 'Cityscapes dataset root folder.')
tf.app.flags.DEFINE_string('output_dir', './tfrecord', 'Path to save converted SSTable of TensorFlow examples.')
_NUM_SHARDS = 10
_FOLDERS_MAP = {'image': 'leftImg8bit', 'label': 'gtFine'}
_POSTFIX_MAP = {'image': '_leftImg8bit', 'label': '_gtFine_labelTrainIds'}
_DATA_FORMAT_MAP = {'image': 'png', 'label': 'png'}
_IMAGE_FILENAME_RE = re.compile('(.+)' + _POSTFIX_MAP['image'])

def _get_files(data, dataset_split):
    if False:
        while True:
            i = 10
    "Gets files for the specified data type and dataset split.\n\n  Args:\n    data: String, desired data ('image' or 'label').\n    dataset_split: String, dataset split ('train', 'val', 'test')\n\n  Returns:\n    A list of sorted file names or None when getting label for\n      test set.\n  "
    if data == 'label' and dataset_split == 'test':
        return None
    pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
    search_files = os.path.join(FLAGS.cityscapes_root, _FOLDERS_MAP[data], dataset_split, '*', pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

def _convert_dataset(dataset_split):
    if False:
        print('Hello World!')
    'Converts the specified dataset split to TFRecord format.\n\n  Args:\n    dataset_split: The dataset split (e.g., train, val).\n\n  Raises:\n    RuntimeError: If loaded image and label have different shape, or if the\n      image file with specified postfix could not be found.\n  '
    image_files = _get_files('image', dataset_split)
    label_files = _get_files('label', dataset_split)
    num_images = len(image_files)
    num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))
    image_reader = build_data.ImageReader('png', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)
    for shard_id in range(_NUM_SHARDS):
        shard_filename = '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS)
        output_filename = os.path.join(FLAGS.output_dir, shard_filename)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i + 1, num_images, shard_id))
                sys.stdout.flush()
                image_data = tf.gfile.FastGFile(image_files[i], 'rb').read()
                (height, width) = image_reader.read_image_dims(image_data)
                seg_data = tf.gfile.FastGFile(label_files[i], 'rb').read()
                (seg_height, seg_width) = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')
                re_match = _IMAGE_FILENAME_RE.search(image_files[i])
                if re_match is None:
                    raise RuntimeError('Invalid image filename: ' + image_files[i])
                filename = os.path.basename(re_match.group(1))
                example = build_data.image_seg_to_tfexample(image_data, filename, height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()

def main(unused_argv):
    if False:
        while True:
            i = 10
    for dataset_split in ['train', 'val']:
        _convert_dataset(dataset_split)
if __name__ == '__main__':
    tf.app.run()