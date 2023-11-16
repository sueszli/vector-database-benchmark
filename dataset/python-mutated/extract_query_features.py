"""Extracts DELF features for query images from Revisited Oxford/Paris datasets.

Note that query images are cropped before feature extraction, as required by the
evaluation protocols of these datasets.

The program checks if descriptors already exist, and skips computation for
those.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import time
import numpy as np
from PIL import Image
from PIL import ImageFile
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import delf_config_pb2
from delf import feature_io
from delf.python.detect_to_retrieve import dataset
from delf import extractor
cmd_args = None
_DELF_EXTENSION = '.delf'
_IMAGE_EXTENSION = '.jpg'
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _PilLoader(path):
    if False:
        print('Hello World!')
    'Helper function to read image with PIL.\n\n  Args:\n    path: Path to image to be loaded.\n\n  Returns:\n    PIL image in RGB format.\n  '
    with tf.gfile.GFile(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def main(argv):
    if False:
        print('Hello World!')
    if len(argv) > 1:
        raise RuntimeError('Too many command-line arguments.')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Reading list of query images and boxes from dataset file...')
    (query_list, _, ground_truth) = dataset.ReadDatasetFile(cmd_args.dataset_file_path)
    num_images = len(query_list)
    tf.logging.info('done! Found %d images', num_images)
    config = delf_config_pb2.DelfConfig()
    with tf.gfile.GFile(cmd_args.delf_config_path, 'r') as f:
        text_format.Merge(f.read(), config)
    if not tf.gfile.Exists(cmd_args.output_features_dir):
        tf.gfile.MakeDirs(cmd_args.output_features_dir)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            extractor_fn = extractor.MakeExtractor(sess, config)
            start = time.clock()
            for i in range(num_images):
                query_image_name = query_list[i]
                input_image_filename = os.path.join(cmd_args.images_dir, query_image_name + _IMAGE_EXTENSION)
                output_feature_filename = os.path.join(cmd_args.output_features_dir, query_image_name + _DELF_EXTENSION)
                if tf.gfile.Exists(output_feature_filename):
                    tf.logging.info('Skipping %s', query_image_name)
                    continue
                bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
                im = np.array(_PilLoader(input_image_filename).crop(bbox))
                (locations_out, descriptors_out, feature_scales_out, attention_out) = extractor_fn(im)
                feature_io.WriteToFile(output_feature_filename, locations_out, feature_scales_out, descriptors_out, attention_out)
            elapsed = time.clock() - start
            print('Processed %d query images in %f seconds' % (num_images, elapsed))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--delf_config_path', type=str, default='/tmp/delf_config_example.pbtxt', help='\n      Path to DelfConfig proto text file with configuration to be used for DELF\n      extraction.\n      ')
    parser.add_argument('--dataset_file_path', type=str, default='/tmp/gnd_roxford5k.mat', help='\n      Dataset file for Revisited Oxford or Paris dataset, in .mat format.\n      ')
    parser.add_argument('--images_dir', type=str, default='/tmp/images', help='\n      Directory where dataset images are located, all in .jpg format.\n      ')
    parser.add_argument('--output_features_dir', type=str, default='/tmp/features', help="\n      Directory where DELF features will be written to. Each image's features\n      will be written to a file with same name, and extension replaced by .delf.\n      ")
    (cmd_args, unparsed) = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)