"""A script that builds boosted trees over higgs data.

If you haven't, please run data_download.py beforehand to prepare the data.

For some more details on this example, please refer to README.md as well.

Note that the model_dir is cleaned up before starting the training.

Usage:
$ python train_higgs.py --n_trees=100 --max_depth=6 --learning_rate=0.1 \\
    --model_dir=/tmp/higgs_model

Note that BoostedTreesClassifier is available since Tensorflow 1.8.0.
So you need to install recent enough version of Tensorflow to use this example.

The training data is by default the first million examples out of 11M examples,
and eval data is by default the last million examples.
They are controlled by --train_start, --train_count, --eval_start, --eval_count.
e.g. to train over the first 10 million examples instead of 1 million:
$ python train_higgs.py --n_trees=100 --max_depth=6 --learning_rate=0.1 \\
    --model_dir=/tmp/higgs_model --train_count=10000000

Training history and metrics can be inspected using tensorboard.
Set --logdir as the --model_dir set by flag when training
(or the default /tmp/higgs_model).
$ tensorboard --logdir=/tmp/higgs_model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from absl import app as absl_app
from absl import flags
import tensorflow as tf
from official.utils.flags import core as flags_core
from official.utils.flags._conventions import help_wrap
from official.utils.logs import logger
NPZ_FILE = 'HIGGS.csv.gz.npz'

def read_higgs_data(data_dir, train_start, train_count, eval_start, eval_count):
    if False:
        for i in range(10):
            print('nop')
    'Reads higgs data from csv and returns train and eval data.\n\n  Args:\n    data_dir: A string, the directory of higgs dataset.\n    train_start: An integer, the start index of train examples within the data.\n    train_count: An integer, the number of train examples within the data.\n    eval_start: An integer, the start index of eval examples within the data.\n    eval_count: An integer, the number of eval examples within the data.\n\n  Returns:\n    Numpy array of train data and eval data.\n  '
    npz_filename = os.path.join(data_dir, NPZ_FILE)
    try:
        with tf.gfile.Open(npz_filename, 'rb') as npz_file:
            with np.load(npz_file) as npz:
                data = npz['data']
    except tf.errors.NotFoundError as e:
        raise RuntimeError('Error loading data; use data_download.py to prepare the data.\n{}: {}'.format(type(e).__name__, e))
    return (data[train_start:train_start + train_count], data[eval_start:eval_start + eval_count])

def make_inputs_from_np_arrays(features_np, label_np):
    if False:
        print('Hello World!')
    "Makes and returns input_fn and feature_columns from numpy arrays.\n\n  The generated input_fn will return tf.data.Dataset of feature dictionary and a\n  label, and feature_columns will consist of the list of\n  tf.feature_column.BucketizedColumn.\n\n  Note, for in-memory training, tf.data.Dataset should contain the whole data\n  as a single tensor. Don't use batch.\n\n  Args:\n    features_np: A numpy ndarray (shape=[batch_size, num_features]) for\n        float32 features.\n    label_np: A numpy ndarray (shape=[batch_size, 1]) for labels.\n\n  Returns:\n    input_fn: A function returning a Dataset of feature dict and label.\n    feature_names: A list of feature names.\n    feature_column: A list of tf.feature_column.BucketizedColumn.\n  "
    num_features = features_np.shape[1]
    features_np_list = np.split(features_np, num_features, axis=1)
    feature_names = ['feature_%02d' % (i + 1) for i in range(num_features)]

    def get_bucket_boundaries(feature):
        if False:
            i = 10
            return i + 15
        'Returns bucket boundaries for feature by percentiles.'
        return np.unique(np.percentile(feature, range(0, 100))).tolist()
    source_columns = [tf.feature_column.numeric_column(feature_name, dtype=tf.float32, default_value=0.0) for feature_name in feature_names]
    bucketized_columns = [tf.feature_column.bucketized_column(source_columns[i], boundaries=get_bucket_boundaries(features_np_list[i])) for i in range(num_features)]

    def input_fn():
        if False:
            while True:
                i = 10
        'Returns features as a dictionary of numpy arrays, and a label.'
        features = {feature_name: tf.constant(features_np_list[i]) for (i, feature_name) in enumerate(feature_names)}
        return tf.data.Dataset.zip((tf.data.Dataset.from_tensors(features), tf.data.Dataset.from_tensors(label_np)))
    return (input_fn, feature_names, bucketized_columns)

def make_eval_inputs_from_np_arrays(features_np, label_np):
    if False:
        while True:
            i = 10
    'Makes eval input as streaming batches.'
    num_features = features_np.shape[1]
    features_np_list = np.split(features_np, num_features, axis=1)
    feature_names = ['feature_%02d' % (i + 1) for i in range(num_features)]

    def input_fn():
        if False:
            while True:
                i = 10
        features = {feature_name: tf.constant(features_np_list[i]) for (i, feature_name) in enumerate(feature_names)}
        return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(features), tf.data.Dataset.from_tensor_slices(label_np))).batch(1000)
    return input_fn

def _make_csv_serving_input_receiver_fn(column_names, column_defaults):
    if False:
        print('Hello World!')
    'Returns serving_input_receiver_fn for csv.\n\n  The input arguments are relevant to `tf.decode_csv()`.\n\n  Args:\n    column_names: a list of column names in the order within input csv.\n    column_defaults: a list of default values with the same size of\n        column_names. Each entity must be either a list of one scalar, or an\n        empty list to denote the corresponding column is required.\n        e.g. [[""], [2.5], []] indicates the third column is required while\n            the first column must be string and the second must be float/double.\n\n  Returns:\n    a serving_input_receiver_fn that handles csv for serving.\n  '

    def serving_input_receiver_fn():
        if False:
            i = 10
            return i + 15
        csv = tf.placeholder(dtype=tf.string, shape=[None], name='csv')
        features = dict(zip(column_names, tf.decode_csv(csv, column_defaults)))
        receiver_tensors = {'inputs': csv}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    return serving_input_receiver_fn

def train_boosted_trees(flags_obj):
    if False:
        i = 10
        return i + 15
    'Train boosted_trees estimator on HIGGS data.\n\n  Args:\n    flags_obj: An object containing parsed flag values.\n  '
    if tf.gfile.Exists(flags_obj.model_dir):
        tf.gfile.DeleteRecursively(flags_obj.model_dir)
    tf.logging.info('## Data loading...')
    (train_data, eval_data) = read_higgs_data(flags_obj.data_dir, flags_obj.train_start, flags_obj.train_count, flags_obj.eval_start, flags_obj.eval_count)
    tf.logging.info('## Data loaded; train: {}{}, eval: {}{}'.format(train_data.dtype, train_data.shape, eval_data.dtype, eval_data.shape))
    (train_input_fn, feature_names, feature_columns) = make_inputs_from_np_arrays(features_np=train_data[:, 1:], label_np=train_data[:, 0:1])
    eval_input_fn = make_eval_inputs_from_np_arrays(features_np=eval_data[:, 1:], label_np=eval_data[:, 0:1])
    tf.logging.info('## Features prepared. Training starts...')
    run_params = {'train_start': flags_obj.train_start, 'train_count': flags_obj.train_count, 'eval_start': flags_obj.eval_start, 'eval_count': flags_obj.eval_count, 'n_trees': flags_obj.n_trees, 'max_depth': flags_obj.max_depth}
    benchmark_logger = logger.config_benchmark_logger(flags_obj)
    benchmark_logger.log_run_info(model_name='boosted_trees', dataset_name='higgs', run_params=run_params, test_id=flags_obj.benchmark_test_id)
    classifier = tf.contrib.estimator.boosted_trees_classifier_train_in_memory(train_input_fn, feature_columns, model_dir=flags_obj.model_dir or None, n_trees=flags_obj.n_trees, max_depth=flags_obj.max_depth, learning_rate=flags_obj.learning_rate)
    eval_results = classifier.evaluate(eval_input_fn)
    benchmark_logger.log_evaluation_result(eval_results)
    if flags_obj.export_dir is not None:
        classifier.export_savedmodel(flags_obj.export_dir, _make_csv_serving_input_receiver_fn(column_names=feature_names, column_defaults=[[0.0]] * len(feature_names)), strip_default_attrs=True)

def main(_):
    if False:
        while True:
            i = 10
    train_boosted_trees(flags.FLAGS)

def define_train_higgs_flags():
    if False:
        return 10
    'Add tree related flags as well as training/eval configuration.'
    flags_core.define_base(clean=False, stop_threshold=False, batch_size=False, num_gpu=False, export_dir=True)
    flags_core.define_benchmark()
    flags.adopt_module_key_flags(flags_core)
    flags.DEFINE_integer(name='train_start', default=0, help=help_wrap('Start index of train examples within the data.'))
    flags.DEFINE_integer(name='train_count', default=1000000, help=help_wrap('Number of train examples within the data.'))
    flags.DEFINE_integer(name='eval_start', default=10000000, help=help_wrap('Start index of eval examples within the data.'))
    flags.DEFINE_integer(name='eval_count', default=1000000, help=help_wrap('Number of eval examples within the data.'))
    flags.DEFINE_integer('n_trees', default=100, help=help_wrap('Number of trees to build.'))
    flags.DEFINE_integer('max_depth', default=6, help=help_wrap('Maximum depths of each tree.'))
    flags.DEFINE_float('learning_rate', default=0.1, help=help_wrap('The learning rate.'))
    flags_core.set_defaults(data_dir='/tmp/higgs_data', model_dir='/tmp/higgs_model')
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_train_higgs_flags()
    absl_app.run(main)