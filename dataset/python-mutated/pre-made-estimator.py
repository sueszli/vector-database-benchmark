from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import pandas as pd
import tensorflow as tf
from optparse import OptionParser
from tensorflow_estimator.python.estimator.canned import prediction_keys
from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.tfpark import TFDataset, TFEstimator
from bigdl.orca.tfpark import ZooOptimizer
from urllib.parse import urlparse
from os.path import exists
from bigdl.dllib.utils import log4Error

def is_local_and_existing_uri(uri):
    if False:
        print('Hello World!')
    parsed_uri = urlparse(uri)
    log4Error.invalidInputError(not parsed_uri.scheme or parsed_uri.scheme == 'file', 'Not Local File!')
    log4Error.invalidInputError(not parsed_uri.netloc or parsed_uri.netloc.lower() == 'localhost', 'Not Local File!')
    log4Error.invalidInputError(exists(parsed_uri.path), 'File Not Exist!')

def make_input_fn(data_df, label_df, mode, batch_size=-1, batch_per_thread=-1):
    if False:
        i = 10
        return i + 15
    if mode == tf.estimator.ModeKeys.TRAIN:

        def input_function():
            if False:
                return 10
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
            ds = ds.shuffle(1000)
            ds = TFDataset.from_tf_data_dataset(dataset=ds, batch_size=batch_size)
            return ds
    elif mode == tf.estimator.ModeKeys.EVAL:

        def input_function():
            if False:
                print('Hello World!')
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
            ds = TFDataset.from_tf_data_dataset(dataset=ds, batch_per_thread=batch_per_thread)
            return ds
    else:

        def input_function():
            if False:
                print('Hello World!')
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df),))
            ds = TFDataset.from_tf_data_dataset(dataset=ds, batch_size=batch_size, batch_per_thread=batch_per_thread)
            return ds
    return input_function
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--data_dir', dest='data_dir')
    (options, args) = parser.parse_args(sys.argv)
    is_local_and_existing_uri(os.path.join(options.data_dir, 'train.csv'))
    is_local_and_existing_uri(os.path.join(options.data_dir, 'eval.csv'))
    dftrain = pd.read_csv(os.path.join(options.data_dir, 'train.csv'))
    dfeval = pd.read_csv(os.path.join(options.data_dir, 'eval.csv'))
    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')
    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
    NUMERIC_COLUMNS = ['age', 'fare']
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = dftrain[feature_name].unique()
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    sc = init_nncontext()
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, optimizer=ZooOptimizer(tf.train.FtrlOptimizer(0.2)), model_dir='/tmp/estimator/linear')
    zoo_est = TFEstimator(linear_est)
    train_input_fn = make_input_fn(dftrain, y_train, mode=tf.estimator.ModeKeys.TRAIN, batch_size=32)
    zoo_est.train(train_input_fn, steps=200)
    eval_input_fn = make_input_fn(dfeval, y_eval, mode=tf.estimator.ModeKeys.EVAL, batch_per_thread=8)
    eval_result = zoo_est.evaluate(eval_input_fn, ['acc'])
    print(eval_result)
    pred_input_fn = make_input_fn(dfeval, y_eval, mode=tf.estimator.ModeKeys.PREDICT, batch_per_thread=8)
    predictions = zoo_est.predict(pred_input_fn, predict_keys=[prediction_keys.PredictionKeys.CLASS_IDS])
    print(predictions.collect())
    print('finished...')
    sc.stop()