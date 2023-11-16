"""Clusters DELF features using the K-means algorithm.

All DELF local feature descriptors for a given dataset's index images are loaded
as the input.

Note that:
-  we only use features extracted from whole images (no features from boxes are
   used).
-  the codebook should be trained on Paris images for Oxford retrieval
   experiments, and vice-versa.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app
from delf import feature_io
from delf.python.detect_to_retrieve import dataset
cmd_args = None
_DELF_EXTENSION = '.delf'
_DELF_DIM = 128
_STATUS_CHECK_ITERATIONS = 100

class _IteratorInitHook(tf.train.SessionRunHook):
    """Hook to initialize data iterator after session is created."""

    def __init__(self):
        if False:
            print('Hello World!')
        super(_IteratorInitHook, self).__init__()
        self.iterator_initializer_fn = None

    def after_create_session(self, session, coord):
        if False:
            print('Hello World!')
        'Initialize the iterator after the session has been created.'
        del coord
        self.iterator_initializer_fn(session)

def main(argv):
    if False:
        print('Hello World!')
    if len(argv) > 1:
        raise RuntimeError('Too many command-line arguments.')
    if tf.gfile.Exists(cmd_args.output_cluster_dir):
        raise RuntimeError('output_cluster_dir = %s already exists. This may indicate that a previous run already wrote checkpoints in this directory, which would lead to incorrect training. Please re-run this script by specifying an inexisting directory.' % cmd_args.output_cluster_dir)
    else:
        tf.gfile.MakeDirs(cmd_args.output_cluster_dir)
    print('Reading list of index images from dataset file...')
    (_, index_list, _) = dataset.ReadDatasetFile(cmd_args.dataset_file_path)
    num_images = len(index_list)
    print('done! Found %d images' % num_images)
    features_for_clustering = []
    start = time.clock()
    print('Starting to collect features from index images...')
    for i in range(num_images):
        if i > 0 and i % _STATUS_CHECK_ITERATIONS == 0:
            elapsed = time.clock() - start
            print('Processing index image %d out of %d, last %d images took %f seconds' % (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
            start = time.clock()
        features_filename = index_list[i] + _DELF_EXTENSION
        features_fullpath = os.path.join(cmd_args.features_dir, features_filename)
        (_, _, features, _, _) = feature_io.ReadFromFile(features_fullpath)
        if features.size != 0:
            assert features.shape[1] == _DELF_DIM
        for feature in features:
            features_for_clustering.append(feature)
    features_for_clustering = np.array(features_for_clustering, dtype=np.float32)
    print('All features were loaded! There are %d features, each with %d dimensions' % (features_for_clustering.shape[0], features_for_clustering.shape[1]))

    def _get_input_fn():
        if False:
            while True:
                i = 10
        'Helper function to create input function and hook for training.\n\n    Returns:\n      input_fn: Input function for k-means Estimator training.\n      init_hook: Hook used to load data during training.\n    '
        init_hook = _IteratorInitHook()

        def _input_fn():
            if False:
                i = 10
                return i + 15
            'Produces tf.data.Dataset object for k-means training.\n\n      Returns:\n        Tensor with the data for training.\n      '
            features_placeholder = tf.placeholder(tf.float32, features_for_clustering.shape)
            delf_dataset = tf.data.Dataset.from_tensor_slices(features_placeholder)
            delf_dataset = delf_dataset.shuffle(1000).batch(features_for_clustering.shape[0])
            iterator = delf_dataset.make_initializable_iterator()

            def _initializer_fn(sess):
                if False:
                    for i in range(10):
                        print('nop')
                'Initialize dataset iterator, feed in the data.'
                sess.run(iterator.initializer, feed_dict={features_placeholder: features_for_clustering})
            init_hook.iterator_initializer_fn = _initializer_fn
            return iterator.get_next()
        return (_input_fn, init_hook)
    (input_fn, init_hook) = _get_input_fn()
    kmeans = tf.estimator.experimental.KMeans(num_clusters=cmd_args.num_clusters, model_dir=cmd_args.output_cluster_dir, use_mini_batch=False)
    print('Starting K-means clustering...')
    start = time.clock()
    for i in range(cmd_args.num_iterations):
        kmeans.train(input_fn, hooks=[init_hook])
        average_sum_squared_error = kmeans.evaluate(input_fn, hooks=[init_hook])['score'] / features_for_clustering.shape[0]
        elapsed = time.clock() - start
        print('K-means iteration %d (out of %d) took %f seconds, average-sum-of-squares: %f' % (i, cmd_args.num_iterations, elapsed, average_sum_squared_error))
        start = time.clock()
    print('K-means clustering finished!')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--dataset_file_path', type=str, default='/tmp/gnd_roxford5k.mat', help='\n      Dataset file for Revisited Oxford or Paris dataset, in .mat format. The\n      list of index images loaded from this file is used to collect local\n      features, which are assumed to be in <image_name>.delf file format.\n      ')
    parser.add_argument('--features_dir', type=str, default='/tmp/features', help='\n      Directory where DELF feature files are to be found.\n      ')
    parser.add_argument('--num_clusters', type=int, default=1024, help='\n      Number of clusters to use.\n      ')
    parser.add_argument('--num_iterations', type=int, default=50, help='\n      Number of iterations to use.\n      ')
    parser.add_argument('--output_cluster_dir', type=str, default='/tmp/cluster', help='\n      Directory where clustering outputs are written to. This directory should\n      not exist before running this script; it will be created during\n      clustering.\n      ')
    (cmd_args, unparsed) = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)