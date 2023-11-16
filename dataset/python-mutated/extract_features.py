"""Extracts DELF features from a list of images, saving them to file.

The images must be in JPG format. The program checks if descriptors already
exist, and skips computation for those.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import time
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import delf_config_pb2
from delf import feature_io
from delf import extractor
cmd_args = None
_DELF_EXT = '.delf'
_STATUS_CHECK_ITERATIONS = 100

def _ReadImageList(list_path):
    if False:
        for i in range(10):
            print('nop')
    'Helper function to read image paths.\n\n  Args:\n    list_path: Path to list of images, one image path per line.\n\n  Returns:\n    image_paths: List of image paths.\n  '
    with tf.gfile.GFile(list_path, 'r') as f:
        image_paths = f.readlines()
    image_paths = [entry.rstrip() for entry in image_paths]
    return image_paths

def main(unused_argv):
    if False:
        i = 10
        return i + 15
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Reading list of images...')
    image_paths = _ReadImageList(cmd_args.list_images_path)
    num_images = len(image_paths)
    tf.logging.info('done! Found %d images', num_images)
    config = delf_config_pb2.DelfConfig()
    with tf.gfile.FastGFile(cmd_args.config_path, 'r') as f:
        text_format.Merge(f.read(), config)
    if not tf.gfile.Exists(cmd_args.output_dir):
        tf.gfile.MakeDirs(cmd_args.output_dir)
    with tf.Graph().as_default():
        filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
        reader = tf.WholeFileReader()
        (_, value) = reader.read(filename_queue)
        image_tf = tf.image.decode_jpeg(value, channels=3)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            extractor_fn = extractor.MakeExtractor(sess, config)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            start = time.clock()
            for i in range(num_images):
                if i == 0:
                    tf.logging.info('Starting to extract DELF features from images...')
                elif i % _STATUS_CHECK_ITERATIONS == 0:
                    elapsed = time.clock() - start
                    tf.logging.info('Processing image %d out of %d, last %d images took %f seconds', i, num_images, _STATUS_CHECK_ITERATIONS, elapsed)
                    start = time.clock()
                im = sess.run(image_tf)
                out_desc_filename = os.path.splitext(os.path.basename(image_paths[i]))[0] + _DELF_EXT
                out_desc_fullpath = os.path.join(cmd_args.output_dir, out_desc_filename)
                if tf.gfile.Exists(out_desc_fullpath):
                    tf.logging.info('Skipping %s', image_paths[i])
                    continue
                (locations_out, descriptors_out, feature_scales_out, attention_out) = extractor_fn(im)
                feature_io.WriteToFile(out_desc_fullpath, locations_out, feature_scales_out, descriptors_out, attention_out)
            coord.request_stop()
            coord.join(threads)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--config_path', type=str, default='delf_config_example.pbtxt', help='\n      Path to DelfConfig proto text file with configuration to be used for DELF\n      extraction.\n      ')
    parser.add_argument('--list_images_path', type=str, default='list_images.txt', help='\n      Path to list of images whose DELF features will be extracted.\n      ')
    parser.add_argument('--output_dir', type=str, default='test_features', help="\n      Directory where DELF features will be written to. Each image's features\n      will be written to a file with same name, and extension replaced by .delf.\n      ")
    (cmd_args, unparsed) = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)