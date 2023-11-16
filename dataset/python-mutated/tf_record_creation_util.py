"""Utilities for creating TFRecords of TF examples for the Open Images dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    if False:
        print('Hello World!')
    'Opens all TFRecord shards for writing and adds them to an exit stack.\n\n  Args:\n    exit_stack: A context2.ExitStack used to automatically closed the TFRecords\n      opened in this function.\n    base_path: The base path for all shards\n    num_shards: The number of shards\n\n  Returns:\n    The list of opened TFRecords. Position k in the list corresponds to shard k.\n  '
    tf_record_output_filenames = ['{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards) for idx in range(num_shards)]
    tfrecords = [exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name)) for file_name in tf_record_output_filenames]
    return tfrecords