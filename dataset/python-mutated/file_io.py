"""Convenience functions for managing dataset file buffers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import atexit
import multiprocessing
import multiprocessing.dummy
import os
import tempfile
import uuid
import numpy as np
import six
import tensorflow as tf

class _GarbageCollector(object):
    """Deletes temporary buffer files at exit.

  Certain tasks (such as NCF Recommendation) require writing buffers to
  temporary files. (Which may be local or distributed.) It is not generally safe
  to delete these files during operation, but they should be cleaned up. This
  class keeps track of temporary files created, and deletes them at exit.
  """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.temp_buffers = []

    def register(self, filepath):
        if False:
            for i in range(10):
                print('nop')
        self.temp_buffers.append(filepath)

    def purge(self):
        if False:
            i = 10
            return i + 15
        try:
            for i in self.temp_buffers:
                if tf.io.gfile.exists(i):
                    tf.io.gfile.remove(i)
                    tf.compat.v1.logging.info('Buffer file {} removed'.format(i))
        except Exception as e:
            tf.compat.v1.logging.error('Failed to cleanup buffer files: {}'.format(e))
_GARBAGE_COLLECTOR = _GarbageCollector()
atexit.register(_GARBAGE_COLLECTOR.purge)
_ROWS_PER_CORE = 50000

def write_to_temp_buffer(dataframe, buffer_folder, columns):
    if False:
        return 10
    if buffer_folder is None:
        (_, buffer_path) = tempfile.mkstemp()
    else:
        tf.io.gfile.makedirs(buffer_folder)
        buffer_path = os.path.join(buffer_folder, str(uuid.uuid4()))
    _GARBAGE_COLLECTOR.register(buffer_path)
    return write_to_buffer(dataframe, buffer_path, columns)

def iter_shard_dataframe(df, rows_per_core=1000):
    if False:
        return 10
    'Two way shard of a dataframe.\n\n  This function evenly shards a dataframe so that it can be mapped efficiently.\n  It yields a list of dataframes with length equal to the number of CPU cores,\n  with each dataframe having rows_per_core rows. (Except for the last batch\n  which may have fewer rows in the dataframes.) Passing vectorized inputs to\n  a pool is more effecient than iterating through a dataframe in serial and\n  passing a list of inputs to the pool.\n\n  Args:\n    df: Pandas dataframe to be sharded.\n    rows_per_core: Number of rows in each shard.\n\n  Returns:\n    A list of dataframe shards.\n  '
    n = len(df)
    num_cores = min([multiprocessing.cpu_count(), n])
    num_blocks = int(np.ceil(n / num_cores / rows_per_core))
    max_batch_size = num_cores * rows_per_core
    for i in range(num_blocks):
        min_index = i * max_batch_size
        max_index = min([(i + 1) * max_batch_size, n])
        df_shard = df[min_index:max_index]
        n_shard = len(df_shard)
        boundaries = np.linspace(0, n_shard, num_cores + 1, dtype=np.int64)
        yield [df_shard[boundaries[j]:boundaries[j + 1]] for j in range(num_cores)]

def _shard_dict_to_examples(shard_dict):
    if False:
        for i in range(10):
            print('nop')
    'Converts a dict of arrays into a list of example bytes.'
    n = [i for i in shard_dict.values()][0].shape[0]
    feature_list = [{} for _ in range(n)]
    for (column, values) in shard_dict.items():
        if len(values.shape) == 1:
            values = np.reshape(values, values.shape + (1,))
        if values.dtype.kind == 'i':
            feature_map = lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=x))
        elif values.dtype.kind == 'f':
            feature_map = lambda x: tf.train.Feature(float_list=tf.train.FloatList(value=x))
        else:
            raise ValueError('Invalid dtype')
        for i in range(n):
            feature_list[i][column] = feature_map(values[i])
    examples = [tf.train.Example(features=tf.train.Features(feature=example_features)) for example_features in feature_list]
    return [e.SerializeToString() for e in examples]

def _serialize_shards(df_shards, columns, pool, writer):
    if False:
        for i in range(10):
            print('nop')
    'Map sharded dataframes to bytes, and write them to a buffer.\n\n  Args:\n    df_shards: A list of pandas dataframes. (Should be of similar size)\n    columns: The dataframe columns to be serialized.\n    pool: A pool to serialize in parallel.\n    writer: A TFRecordWriter to write the serialized shards.\n  '
    map_inputs = [{c: np.stack(shard[c].values, axis=0) for c in columns} for shard in df_shards]
    for inp in map_inputs:
        assert len(set([v.shape[0] for v in inp.values()])) == 1
        for val in inp.values():
            assert hasattr(val, 'dtype')
            assert hasattr(val.dtype, 'kind')
            assert val.dtype.kind in ('i', 'f')
            assert len(val.shape) in (1, 2)
    shard_bytes = pool.map(_shard_dict_to_examples, map_inputs)
    for s in shard_bytes:
        for example in s:
            writer.write(example)

def write_to_buffer(dataframe, buffer_path, columns, expected_size=None):
    if False:
        print('Hello World!')
    'Write a dataframe to a binary file for a dataset to consume.\n\n  Args:\n    dataframe: The pandas dataframe to be serialized.\n    buffer_path: The path where the serialized results will be written.\n    columns: The dataframe columns to be serialized.\n    expected_size: The size in bytes of the serialized results. This is used to\n      lazily construct the buffer.\n\n  Returns:\n    The path of the buffer.\n  '
    if tf.io.gfile.exists(buffer_path) and tf.io.gfile.stat(buffer_path).length > 0:
        actual_size = tf.io.gfile.stat(buffer_path).length
        if expected_size == actual_size:
            return buffer_path
        tf.compat.v1.logging.warning('Existing buffer {} has size {}. Expected size {}. Deleting and rebuilding buffer.'.format(buffer_path, actual_size, expected_size))
        tf.io.gfile.remove(buffer_path)
    if dataframe is None:
        raise ValueError('dataframe was None but a valid existing buffer was not found.')
    tf.io.gfile.makedirs(os.path.split(buffer_path)[0])
    tf.compat.v1.logging.info('Constructing TFRecordDataset buffer: {}'.format(buffer_path))
    count = 0
    pool = multiprocessing.dummy.Pool(multiprocessing.cpu_count())
    try:
        with tf.io.TFRecordWriter(buffer_path) as writer:
            for df_shards in iter_shard_dataframe(df=dataframe, rows_per_core=_ROWS_PER_CORE):
                _serialize_shards(df_shards, columns, pool, writer)
                count += sum([len(s) for s in df_shards])
                tf.compat.v1.logging.info('{}/{} examples written.'.format(str(count).ljust(8), len(dataframe)))
    finally:
        pool.terminate()
    tf.compat.v1.logging.info('Buffer write complete.')
    return buffer_path