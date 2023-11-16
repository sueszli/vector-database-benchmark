"""Generate a synthetic dataset."""
import os
import numpy as np
from six.moves import xrange
import tensorflow as tf
import synthetic_model
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_dir', None, 'Directory where to write the dataset and the configs.')
tf.app.flags.DEFINE_integer('count', 1000, 'Number of samples to generate.')

def int64_feature(values):
    if False:
        for i in range(10):
            print('nop')
    'Returns a TF-Feature of int64s.\n\n  Args:\n    values: A scalar or list of values.\n\n  Returns:\n    A TF-Feature.\n  '
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def float_feature(values):
    if False:
        return 10
    'Returns a TF-Feature of floats.\n\n  Args:\n    values: A scalar of list of values.\n\n  Returns:\n    A TF-Feature.\n  '
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def AddToTFRecord(code, tfrecord_writer):
    if False:
        print('Hello World!')
    example = tf.train.Example(features=tf.train.Features(feature={'code_shape': int64_feature(code.shape), 'code': float_feature(code.flatten().tolist())}))
    tfrecord_writer.write(example.SerializeToString())

def GenerateDataset(filename, count, code_shape):
    if False:
        while True:
            i = 10
    with tf.python_io.TFRecordWriter(filename) as tfrecord_writer:
        for _ in xrange(count):
            code = synthetic_model.GenerateSingleCode(code_shape)
            code = 2.0 * code - 1.0
            AddToTFRecord(code, tfrecord_writer)

def main(argv=None):
    if False:
        print('Hello World!')
    GenerateDataset(os.path.join(FLAGS.dataset_dir + '/synthetic_dataset'), FLAGS.count, [35, 48, 8])
if __name__ == '__main__':
    tf.app.run()