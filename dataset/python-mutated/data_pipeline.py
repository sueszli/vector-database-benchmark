"""Asynchronous data producer for the NCF pipeline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import atexit
import functools
import os
import sys
import tempfile
import threading
import time
import timeit
import traceback
import typing
import numpy as np
import six
from six.moves import queue
import tensorflow as tf
from absl import logging
from official.recommendation import constants as rconst
from official.recommendation import movielens
from official.recommendation import popen_helper
from official.recommendation import stat_utils
SUMMARY_TEMPLATE = 'General:\n{spacer}Num users: {num_users}\n{spacer}Num items: {num_items}\n\nTraining:\n{spacer}Positive count:          {train_pos_ct}\n{spacer}Batch size:              {train_batch_size} {multiplier}\n{spacer}Batch count per epoch:   {train_batch_ct}\n\nEval:\n{spacer}Positive count:          {eval_pos_ct}\n{spacer}Batch size:              {eval_batch_size} {multiplier}\n{spacer}Batch count per epoch:   {eval_batch_ct}'

class DatasetManager(object):
    """Helper class for handling TensorFlow specific data tasks.

  This class takes the (relatively) framework agnostic work done by the data
  constructor classes and handles the TensorFlow specific portions (TFRecord
  management, tf.Dataset creation, etc.).
  """

    def __init__(self, is_training, stream_files, batches_per_epoch, shard_root=None, deterministic=False, num_train_epochs=None):
        if False:
            return 10
        'Constructs a `DatasetManager` instance.\n    Args:\n      is_training: Boolean of whether the data provided is training or\n        evaluation data. This determines whether to reuse the data\n        (if is_training=False) and the exact structure to use when storing and\n        yielding data.\n      stream_files: Boolean indicating whether data should be serialized and\n        written to file shards.\n      batches_per_epoch: The number of batches in a single epoch.\n      shard_root: The base directory to be used when stream_files=True.\n      deterministic: Forgo non-deterministic speedups. (i.e. sloppy=True)\n      num_train_epochs: Number of epochs to generate. If None, then each\n        call to `get_dataset()` increments the number of epochs requested.\n    '
        self._is_training = is_training
        self._deterministic = deterministic
        self._stream_files = stream_files
        self._writers = []
        self._write_locks = [threading.RLock() for _ in range(rconst.NUM_FILE_SHARDS)] if stream_files else []
        self._batches_per_epoch = batches_per_epoch
        self._epochs_completed = 0
        self._epochs_requested = num_train_epochs if num_train_epochs else 0
        self._shard_root = shard_root
        self._result_queue = queue.Queue()
        self._result_reuse = []

    @property
    def current_data_root(self):
        if False:
            print('Hello World!')
        subdir = rconst.TRAIN_FOLDER_TEMPLATE.format(self._epochs_completed) if self._is_training else rconst.EVAL_FOLDER
        return os.path.join(self._shard_root, subdir)

    def buffer_reached(self):
        if False:
            i = 10
            return i + 15
        return self._epochs_completed - self._epochs_requested >= rconst.CYCLES_TO_BUFFER and self._is_training

    @staticmethod
    def serialize(data):
        if False:
            print('Hello World!')
        'Convert NumPy arrays into a TFRecords entry.'

        def create_int_feature(values):
            if False:
                print('Hello World!')
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        feature_dict = {k: create_int_feature(v.astype(np.int64)) for (k, v) in data.items()}
        return tf.train.Example(features=tf.train.Features(feature=feature_dict)).SerializeToString()

    @staticmethod
    def deserialize(serialized_data, batch_size=None, is_training=True):
        if False:
            for i in range(10):
                print('nop')
        'Convert serialized TFRecords into tensors.\n\n    Args:\n      serialized_data: A tensor containing serialized records.\n      batch_size: The data arrives pre-batched, so batch size is needed to\n        deserialize the data.\n      is_training: Boolean, whether data to deserialize to training data\n        or evaluation data.\n    '

        def _get_feature_map(batch_size, is_training=True):
            if False:
                i = 10
                return i + 15
            'Returns data format of the serialized tf record file.'
            if is_training:
                return {movielens.USER_COLUMN: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64), movielens.ITEM_COLUMN: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64), rconst.VALID_POINT_MASK: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64), 'labels': tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64)}
            else:
                return {movielens.USER_COLUMN: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64), movielens.ITEM_COLUMN: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64), rconst.DUPLICATE_MASK: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64)}
        features = tf.io.parse_single_example(serialized_data, _get_feature_map(batch_size, is_training=is_training))
        users = tf.cast(features[movielens.USER_COLUMN], rconst.USER_DTYPE)
        items = tf.cast(features[movielens.ITEM_COLUMN], rconst.ITEM_DTYPE)
        if is_training:
            valid_point_mask = tf.cast(features[rconst.VALID_POINT_MASK], tf.bool)
            fake_dup_mask = tf.zeros_like(users)
            return {movielens.USER_COLUMN: users, movielens.ITEM_COLUMN: items, rconst.VALID_POINT_MASK: valid_point_mask, rconst.TRAIN_LABEL_KEY: tf.reshape(tf.cast(features['labels'], tf.bool), (batch_size, 1)), rconst.DUPLICATE_MASK: fake_dup_mask}
        else:
            labels = tf.cast(tf.zeros_like(users), tf.bool)
            fake_valid_pt_mask = tf.cast(tf.zeros_like(users), tf.bool)
            return {movielens.USER_COLUMN: users, movielens.ITEM_COLUMN: items, rconst.DUPLICATE_MASK: tf.cast(features[rconst.DUPLICATE_MASK], tf.bool), rconst.VALID_POINT_MASK: fake_valid_pt_mask, rconst.TRAIN_LABEL_KEY: labels}

    def put(self, index, data):
        if False:
            return 10
        'Store data for later consumption.\n\n    Because there are several paths for storing and yielding data (queues,\n    lists, files) the data producer simply provides the data in a standard\n    format at which point the dataset manager handles storing it in the correct\n    form.\n\n    Args:\n      index: Used to select shards when writing to files.\n      data: A dict of the data to be stored. This method mutates data, and\n        therefore expects to be the only consumer.\n    '
        if self._is_training:
            mask_start_index = data.pop(rconst.MASK_START_INDEX)
            batch_size = data[movielens.ITEM_COLUMN].shape[0]
            data[rconst.VALID_POINT_MASK] = np.expand_dims(np.less(np.arange(batch_size), mask_start_index), -1)
        if self._stream_files:
            example_bytes = self.serialize(data)
            with self._write_locks[index % rconst.NUM_FILE_SHARDS]:
                self._writers[index % rconst.NUM_FILE_SHARDS].write(example_bytes)
        else:
            self._result_queue.put((data, data.pop('labels')) if self._is_training else data)

    def start_construction(self):
        if False:
            for i in range(10):
                print('nop')
        if self._stream_files:
            tf.io.gfile.makedirs(self.current_data_root)
            template = os.path.join(self.current_data_root, rconst.SHARD_TEMPLATE)
            self._writers = [tf.io.TFRecordWriter(template.format(i)) for i in range(rconst.NUM_FILE_SHARDS)]

    def end_construction(self):
        if False:
            print('Hello World!')
        if self._stream_files:
            [writer.close() for writer in self._writers]
            self._writers = []
            self._result_queue.put(self.current_data_root)
        self._epochs_completed += 1

    def data_generator(self, epochs_between_evals):
        if False:
            return 10
        'Yields examples during local training.'
        assert not self._stream_files
        assert self._is_training or epochs_between_evals == 1
        if self._is_training:
            for _ in range(self._batches_per_epoch * epochs_between_evals):
                yield self._result_queue.get(timeout=300)
        elif self._result_reuse:
            assert len(self._result_reuse) == self._batches_per_epoch
            for i in self._result_reuse:
                yield i
        else:
            for _ in range(self._batches_per_epoch * epochs_between_evals):
                result = self._result_queue.get(timeout=300)
                self._result_reuse.append(result)
                yield result

    def increment_request_epoch(self):
        if False:
            while True:
                i = 10
        self._epochs_requested += 1

    def get_dataset(self, batch_size, epochs_between_evals):
        if False:
            print('Hello World!')
        'Construct the dataset to be used for training and eval.\n\n    For local training, data is provided through Dataset.from_generator. For\n    remote training (TPUs) the data is first serialized to files and then sent\n    to the TPU through a StreamingFilesDataset.\n\n    Args:\n      batch_size: The per-replica batch size of the dataset.\n      epochs_between_evals: How many epochs worth of data to yield.\n        (Generator mode only.)\n    '
        self.increment_request_epoch()
        if self._stream_files:
            if epochs_between_evals > 1:
                raise ValueError('epochs_between_evals > 1 not supported for file based dataset.')
            epoch_data_dir = self._result_queue.get(timeout=300)
            if not self._is_training:
                self._result_queue.put(epoch_data_dir)
            file_pattern = os.path.join(epoch_data_dir, rconst.SHARD_TEMPLATE.format('*'))
            from tensorflow.contrib.tpu.python.tpu.datasets import StreamingFilesDataset
            dataset = StreamingFilesDataset(files=file_pattern, worker_job=popen_helper.worker_job(), num_parallel_reads=rconst.NUM_FILE_SHARDS, num_epochs=1, sloppy=not self._deterministic)
            map_fn = functools.partial(self.deserialize, batch_size=batch_size, is_training=self._is_training)
            dataset = dataset.map(map_fn, num_parallel_calls=16)
        else:
            types = {movielens.USER_COLUMN: rconst.USER_DTYPE, movielens.ITEM_COLUMN: rconst.ITEM_DTYPE}
            shapes = {movielens.USER_COLUMN: tf.TensorShape([batch_size, 1]), movielens.ITEM_COLUMN: tf.TensorShape([batch_size, 1])}
            if self._is_training:
                types[rconst.VALID_POINT_MASK] = np.bool
                shapes[rconst.VALID_POINT_MASK] = tf.TensorShape([batch_size, 1])
                types = (types, np.bool)
                shapes = (shapes, tf.TensorShape([batch_size, 1]))
            else:
                types[rconst.DUPLICATE_MASK] = np.bool
                shapes[rconst.DUPLICATE_MASK] = tf.TensorShape([batch_size, 1])
            data_generator = functools.partial(self.data_generator, epochs_between_evals=epochs_between_evals)
            dataset = tf.data.Dataset.from_generator(generator=data_generator, output_types=types, output_shapes=shapes)
        return dataset.prefetch(16)

    def make_input_fn(self, batch_size):
        if False:
            i = 10
            return i + 15
        'Create an input_fn which checks for batch size consistency.'

        def input_fn(params):
            if False:
                while True:
                    i = 10
            'Returns batches for training.'
            param_batch_size = params['batch_size'] if self._is_training else params.get('eval_batch_size') or params['batch_size']
            if batch_size != param_batch_size:
                raise ValueError('producer batch size ({}) differs from params batch size ({})'.format(batch_size, param_batch_size))
            epochs_between_evals = params.get('epochs_between_evals', 1) if self._is_training else 1
            return self.get_dataset(batch_size=batch_size, epochs_between_evals=epochs_between_evals)
        return input_fn

class BaseDataConstructor(threading.Thread):
    """Data constructor base class.

  This class manages the control flow for constructing data. It is not meant
  to be used directly, but instead subclasses should implement the following
  two methods:

    self.construct_lookup_variables
    self.lookup_negative_items

  """

    def __init__(self, maximum_number_epochs, num_users, num_items, user_map, item_map, train_pos_users, train_pos_items, train_batch_size, batches_per_train_step, num_train_negatives, eval_pos_users, eval_pos_items, eval_batch_size, batches_per_eval_step, stream_files, deterministic=False, epoch_dir=None, num_train_epochs=None, create_data_offline=False):
        if False:
            i = 10
            return i + 15
        self._maximum_number_epochs = maximum_number_epochs
        self._num_users = num_users
        self._num_items = num_items
        self.user_map = user_map
        self.item_map = item_map
        self._train_pos_users = train_pos_users
        self._train_pos_items = train_pos_items
        self.train_batch_size = train_batch_size
        self._num_train_negatives = num_train_negatives
        self._batches_per_train_step = batches_per_train_step
        self._eval_pos_users = eval_pos_users
        self._eval_pos_items = eval_pos_items
        self.eval_batch_size = eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.create_data_offline = create_data_offline
        if self._train_pos_users.shape != self._train_pos_items.shape:
            raise ValueError('User positives ({}) is different from item positives ({})'.format(self._train_pos_users.shape, self._train_pos_items.shape))
        (self._train_pos_count,) = self._train_pos_users.shape
        self._elements_in_epoch = (1 + num_train_negatives) * self._train_pos_count
        self.train_batches_per_epoch = self._count_batches(self._elements_in_epoch, train_batch_size, batches_per_train_step)
        if eval_batch_size % (1 + rconst.NUM_EVAL_NEGATIVES):
            raise ValueError('Eval batch size {} is not divisible by {}'.format(eval_batch_size, 1 + rconst.NUM_EVAL_NEGATIVES))
        self._eval_users_per_batch = int(eval_batch_size // (1 + rconst.NUM_EVAL_NEGATIVES))
        self._eval_elements_in_epoch = num_users * (1 + rconst.NUM_EVAL_NEGATIVES)
        self.eval_batches_per_epoch = self._count_batches(self._eval_elements_in_epoch, eval_batch_size, batches_per_eval_step)
        self._current_epoch_order = np.empty(shape=(0,))
        self._shuffle_iterator = None
        self._shuffle_with_forkpool = not stream_files
        if stream_files:
            self._shard_root = epoch_dir or tempfile.mkdtemp(prefix='ncf_')
            atexit.register(tf.io.gfile.rmtree, dirname=self._shard_root)
        else:
            self._shard_root = None
        self._train_dataset = DatasetManager(True, stream_files, self.train_batches_per_epoch, self._shard_root, deterministic, num_train_epochs)
        self._eval_dataset = DatasetManager(False, stream_files, self.eval_batches_per_epoch, self._shard_root, deterministic, num_train_epochs)
        super(BaseDataConstructor, self).__init__()
        self.daemon = True
        self._stop_loop = False
        self._fatal_exception = None
        self.deterministic = deterministic

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        multiplier = '(x{} devices)'.format(self._batches_per_train_step) if self._batches_per_train_step > 1 else ''
        summary = SUMMARY_TEMPLATE.format(spacer='  ', num_users=self._num_users, num_items=self._num_items, train_pos_ct=self._train_pos_count, train_batch_size=self.train_batch_size, train_batch_ct=self.train_batches_per_epoch, eval_pos_ct=self._num_users, eval_batch_size=self.eval_batch_size, eval_batch_ct=self.eval_batches_per_epoch, multiplier=multiplier)
        return super(BaseDataConstructor, self).__str__() + '\n' + summary

    @staticmethod
    def _count_batches(example_count, batch_size, batches_per_step):
        if False:
            print('Hello World!')
        'Determine the number of batches, rounding up to fill all devices.'
        x = (example_count + batch_size - 1) // batch_size
        return (x + batches_per_step - 1) // batches_per_step * batches_per_step

    def stop_loop(self):
        if False:
            i = 10
            return i + 15
        self._stop_loop = True

    def construct_lookup_variables(self):
        if False:
            for i in range(10):
                print('nop')
        'Perform any one time pre-compute work.'
        raise NotImplementedError

    def lookup_negative_items(self, **kwargs):
        if False:
            return 10
        'Randomly sample negative items for given users.'
        raise NotImplementedError

    def _run(self):
        if False:
            i = 10
            return i + 15
        atexit.register(self.stop_loop)
        self._start_shuffle_iterator()
        self.construct_lookup_variables()
        self._construct_training_epoch()
        self._construct_eval_epoch()
        for _ in range(self._maximum_number_epochs - 1):
            self._construct_training_epoch()
        self.stop_loop()

    def run(self):
        if False:
            return 10
        try:
            self._run()
        except Exception as e:
            traceback.print_exc()
            self._fatal_exception = e
            sys.stderr.flush()
            raise

    def _start_shuffle_iterator(self):
        if False:
            i = 10
            return i + 15
        if self._shuffle_with_forkpool:
            pool = popen_helper.get_forkpool(3, closing=False)
        else:
            pool = popen_helper.get_threadpool(1, closing=False)
        atexit.register(pool.close)
        args = [(self._elements_in_epoch, stat_utils.random_int32()) for _ in range(self._maximum_number_epochs)]
        imap = pool.imap if self.deterministic else pool.imap_unordered
        self._shuffle_iterator = imap(stat_utils.permutation, args)

    def _get_training_batch(self, i):
        if False:
            while True:
                i = 10
        'Construct a single batch of training data.\n\n    Args:\n      i: The index of the batch. This is used when stream_files=True to assign\n        data to file shards.\n    '
        batch_indices = self._current_epoch_order[i * self.train_batch_size:(i + 1) * self.train_batch_size]
        (mask_start_index,) = batch_indices.shape
        batch_ind_mod = np.mod(batch_indices, self._train_pos_count)
        users = self._train_pos_users[batch_ind_mod]
        negative_indices = np.greater_equal(batch_indices, self._train_pos_count)
        negative_users = users[negative_indices]
        negative_items = self.lookup_negative_items(negative_users=negative_users)
        items = self._train_pos_items[batch_ind_mod]
        items[negative_indices] = negative_items
        labels = np.logical_not(negative_indices)
        pad_length = self.train_batch_size - mask_start_index
        if pad_length:
            user_pad = np.arange(pad_length, dtype=users.dtype) % self._num_users
            item_pad = np.arange(pad_length, dtype=items.dtype) % self._num_items
            label_pad = np.zeros(shape=(pad_length,), dtype=labels.dtype)
            users = np.concatenate([users, user_pad])
            items = np.concatenate([items, item_pad])
            labels = np.concatenate([labels, label_pad])
        self._train_dataset.put(i, {movielens.USER_COLUMN: np.reshape(users, (self.train_batch_size, 1)), movielens.ITEM_COLUMN: np.reshape(items, (self.train_batch_size, 1)), rconst.MASK_START_INDEX: np.array(mask_start_index, dtype=np.int32), 'labels': np.reshape(labels, (self.train_batch_size, 1))})

    def _wait_to_construct_train_epoch(self):
        if False:
            i = 10
            return i + 15
        count = 0
        while self._train_dataset.buffer_reached() and (not self._stop_loop):
            time.sleep(0.01)
            count += 1
            if count >= 100 and np.log10(count) == np.round(np.log10(count)):
                logging.info('Waited {} times for training data to be consumed'.format(count))

    def _construct_training_epoch(self):
        if False:
            while True:
                i = 10
        'Loop to construct a batch of training data.'
        if not self.create_data_offline:
            self._wait_to_construct_train_epoch()
        start_time = timeit.default_timer()
        if self._stop_loop:
            return
        self._train_dataset.start_construction()
        map_args = list(range(self.train_batches_per_epoch))
        self._current_epoch_order = next(self._shuffle_iterator)
        get_pool = popen_helper.get_fauxpool if self.deterministic else popen_helper.get_threadpool
        with get_pool(6) as pool:
            pool.map(self._get_training_batch, map_args)
        self._train_dataset.end_construction()
        logging.info('Epoch construction complete. Time: {:.1f} seconds'.format(timeit.default_timer() - start_time))

    @staticmethod
    def _assemble_eval_batch(users, positive_items, negative_items, users_per_batch):
        if False:
            i = 10
            return i + 15
        'Construct duplicate_mask and structure data accordingly.\n\n    The positive items should be last so that they lose ties. However, they\n    should not be masked out if the true eval positive happens to be\n    selected as a negative. So instead, the positive is placed in the first\n    position, and then switched with the last element after the duplicate\n    mask has been computed.\n\n    Args:\n      users: An array of users in a batch. (should be identical along axis 1)\n      positive_items: An array (batch_size x 1) of positive item indices.\n      negative_items: An array of negative item indices.\n      users_per_batch: How many users should be in the batch. This is passed\n        as an argument so that ncf_test.py can use this method.\n\n    Returns:\n      User, item, and duplicate_mask arrays.\n    '
        items = np.concatenate([positive_items, negative_items], axis=1)
        if users.shape[0] < users_per_batch:
            pad_rows = users_per_batch - users.shape[0]
            padding = np.zeros(shape=(pad_rows, users.shape[1]), dtype=np.int32)
            users = np.concatenate([users, padding.astype(users.dtype)], axis=0)
            items = np.concatenate([items, padding.astype(items.dtype)], axis=0)
        duplicate_mask = stat_utils.mask_duplicates(items, axis=1).astype(np.bool)
        items[:, (0, -1)] = items[:, (-1, 0)]
        duplicate_mask[:, (0, -1)] = duplicate_mask[:, (-1, 0)]
        assert users.shape == items.shape == duplicate_mask.shape
        return (users, items, duplicate_mask)

    def _get_eval_batch(self, i):
        if False:
            print('Hello World!')
        'Construct a single batch of evaluation data.\n\n    Args:\n      i: The index of the batch.\n    '
        low_index = i * self._eval_users_per_batch
        high_index = (i + 1) * self._eval_users_per_batch
        users = np.repeat(self._eval_pos_users[low_index:high_index, np.newaxis], 1 + rconst.NUM_EVAL_NEGATIVES, axis=1)
        positive_items = self._eval_pos_items[low_index:high_index, np.newaxis]
        negative_items = self.lookup_negative_items(negative_users=users[:, :-1]).reshape(-1, rconst.NUM_EVAL_NEGATIVES)
        (users, items, duplicate_mask) = self._assemble_eval_batch(users, positive_items, negative_items, self._eval_users_per_batch)
        self._eval_dataset.put(i, {movielens.USER_COLUMN: np.reshape(users.flatten(), (self.eval_batch_size, 1)), movielens.ITEM_COLUMN: np.reshape(items.flatten(), (self.eval_batch_size, 1)), rconst.DUPLICATE_MASK: np.reshape(duplicate_mask.flatten(), (self.eval_batch_size, 1))})

    def _construct_eval_epoch(self):
        if False:
            for i in range(10):
                print('nop')
        'Loop to construct data for evaluation.'
        if self._stop_loop:
            return
        start_time = timeit.default_timer()
        self._eval_dataset.start_construction()
        map_args = [i for i in range(self.eval_batches_per_epoch)]
        get_pool = popen_helper.get_fauxpool if self.deterministic else popen_helper.get_threadpool
        with get_pool(6) as pool:
            pool.map(self._get_eval_batch, map_args)
        self._eval_dataset.end_construction()
        logging.info('Eval construction complete. Time: {:.1f} seconds'.format(timeit.default_timer() - start_time))

    def make_input_fn(self, is_training):
        if False:
            while True:
                i = 10
        if self._fatal_exception is not None:
            raise ValueError('Fatal exception in the data production loop: {}'.format(self._fatal_exception))
        return self._train_dataset.make_input_fn(self.train_batch_size) if is_training else self._eval_dataset.make_input_fn(self.eval_batch_size)

    def increment_request_epoch(self):
        if False:
            return 10
        self._train_dataset.increment_request_epoch()

class DummyConstructor(threading.Thread):
    """Class for running with synthetic data."""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(DummyConstructor, self).__init__(*args, **kwargs)
        self.train_batches_per_epoch = rconst.SYNTHETIC_BATCHES_PER_EPOCH
        self.eval_batches_per_epoch = rconst.SYNTHETIC_BATCHES_PER_EPOCH

    def run(self):
        if False:
            return 10
        pass

    def stop_loop(self):
        if False:
            return 10
        pass

    def increment_request_epoch(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    def make_input_fn(is_training):
        if False:
            for i in range(10):
                print('nop')
        'Construct training input_fn that uses synthetic data.'

        def input_fn(params):
            if False:
                for i in range(10):
                    print('nop')
            'Returns dummy input batches for training.'
            batch_size = params['batch_size'] if is_training else params.get('eval_batch_size') or params['batch_size']
            num_users = params['num_users']
            num_items = params['num_items']
            users = tf.random.uniform([batch_size, 1], dtype=tf.int32, minval=0, maxval=num_users)
            items = tf.random.uniform([batch_size, 1], dtype=tf.int32, minval=0, maxval=num_items)
            if is_training:
                valid_point_mask = tf.cast(tf.random.uniform([batch_size, 1], dtype=tf.int32, minval=0, maxval=2), tf.bool)
                labels = tf.cast(tf.random.uniform([batch_size, 1], dtype=tf.int32, minval=0, maxval=2), tf.bool)
                data = ({movielens.USER_COLUMN: users, movielens.ITEM_COLUMN: items, rconst.VALID_POINT_MASK: valid_point_mask}, labels)
            else:
                dupe_mask = tf.cast(tf.random.uniform([batch_size, 1], dtype=tf.int32, minval=0, maxval=2), tf.bool)
                data = {movielens.USER_COLUMN: users, movielens.ITEM_COLUMN: items, rconst.DUPLICATE_MASK: dupe_mask}
            dataset = tf.data.Dataset.from_tensors(data).repeat(rconst.SYNTHETIC_BATCHES_PER_EPOCH * params['batches_per_step'])
            dataset = dataset.prefetch(32)
            return dataset
        return input_fn

class MaterializedDataConstructor(BaseDataConstructor):
    """Materialize a table of negative examples for fast negative generation.

  This class creates a table (num_users x num_items) containing all of the
  negative examples for each user. This table is conceptually ragged; that is to
  say the items dimension will have a number of unused elements at the end equal
  to the number of positive elements for a given user. For instance:

  num_users = 3
  num_items = 5
  positives = [[1, 3], [0], [1, 2, 3, 4]]

  will generate a negative table:
  [
    [0         2         4         int32max  int32max],
    [1         2         3         4         int32max],
    [0         int32max  int32max  int32max  int32max],
  ]

  and a vector of per-user negative counts, which in this case would be:
    [3, 4, 1]

  When sampling negatives, integers are (nearly) uniformly selected from the
  range [0, per_user_neg_count[user]) which gives a column_index, at which
  point the negative can be selected as:
    negative_table[user, column_index]

  This technique will not scale; however MovieLens is small enough that even
  a pre-compute which is quadratic in problem size will still fit in memory. A
  more scalable lookup method is in the works.
  """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(MaterializedDataConstructor, self).__init__(*args, **kwargs)
        self._negative_table = None
        self._per_user_neg_count = None

    def construct_lookup_variables(self):
        if False:
            print('Hello World!')
        start_time = timeit.default_timer()
        inner_bounds = np.argwhere(self._train_pos_users[1:] - self._train_pos_users[:-1])[:, 0] + 1
        (upper_bound,) = self._train_pos_users.shape
        index_bounds = [0] + inner_bounds.tolist() + [upper_bound]
        self._negative_table = np.zeros(shape=(self._num_users, self._num_items), dtype=rconst.ITEM_DTYPE)
        self._negative_table += np.iinfo(rconst.ITEM_DTYPE).max
        assert self._num_items < np.iinfo(rconst.ITEM_DTYPE).max
        full_set = np.arange(self._num_items, dtype=rconst.ITEM_DTYPE)
        self._per_user_neg_count = np.zeros(shape=(self._num_users,), dtype=np.int32)
        for i in range(self._num_users):
            positives = self._train_pos_items[index_bounds[i]:index_bounds[i + 1]]
            negatives = np.delete(full_set, positives)
            self._per_user_neg_count[i] = self._num_items - positives.shape[0]
            self._negative_table[i, :self._per_user_neg_count[i]] = negatives
        logging.info('Negative sample table built. Time: {:.1f} seconds'.format(timeit.default_timer() - start_time))

    def lookup_negative_items(self, negative_users, **kwargs):
        if False:
            while True:
                i = 10
        negative_item_choice = stat_utils.very_slightly_biased_randint(self._per_user_neg_count[negative_users])
        return self._negative_table[negative_users, negative_item_choice]

class BisectionDataConstructor(BaseDataConstructor):
    """Use bisection to index within positive examples.

  This class tallies the number of negative items which appear before each
  positive item for a user. This means that in order to select the ith negative
  item for a user, it only needs to determine which two positive items bound
  it at which point the item id for the ith negative is a simply algebraic
  expression.
  """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(BisectionDataConstructor, self).__init__(*args, **kwargs)
        self.index_bounds = None
        self._sorted_train_pos_items = None
        self._total_negatives = None

    def _index_segment(self, user):
        if False:
            return 10
        (lower, upper) = self.index_bounds[user:user + 2]
        items = self._sorted_train_pos_items[lower:upper]
        negatives_since_last_positive = np.concatenate([items[0][np.newaxis], items[1:] - items[:-1] - 1])
        return np.cumsum(negatives_since_last_positive)

    def construct_lookup_variables(self):
        if False:
            return 10
        start_time = timeit.default_timer()
        inner_bounds = np.argwhere(self._train_pos_users[1:] - self._train_pos_users[:-1])[:, 0] + 1
        (upper_bound,) = self._train_pos_users.shape
        self.index_bounds = np.array([0] + inner_bounds.tolist() + [upper_bound])
        assert np.array_equal(self._train_pos_users[self.index_bounds[:-1]], np.arange(self._num_users))
        self._sorted_train_pos_items = self._train_pos_items.copy()
        for i in range(self._num_users):
            (lower, upper) = self.index_bounds[i:i + 2]
            self._sorted_train_pos_items[lower:upper].sort()
        self._total_negatives = np.concatenate([self._index_segment(i) for i in range(self._num_users)])
        logging.info('Negative total vector built. Time: {:.1f} seconds'.format(timeit.default_timer() - start_time))

    def lookup_negative_items(self, negative_users, **kwargs):
        if False:
            while True:
                i = 10
        output = np.zeros(shape=negative_users.shape, dtype=rconst.ITEM_DTYPE) - 1
        left_index = self.index_bounds[negative_users]
        right_index = self.index_bounds[negative_users + 1] - 1
        num_positives = right_index - left_index + 1
        num_negatives = self._num_items - num_positives
        neg_item_choice = stat_utils.very_slightly_biased_randint(num_negatives)
        use_shortcut = neg_item_choice >= self._total_negatives[right_index]
        output[use_shortcut] = (self._sorted_train_pos_items[right_index] + 1 + (neg_item_choice - self._total_negatives[right_index]))[use_shortcut]
        if np.all(use_shortcut):
            return output
        not_use_shortcut = np.logical_not(use_shortcut)
        left_index = left_index[not_use_shortcut]
        right_index = right_index[not_use_shortcut]
        neg_item_choice = neg_item_choice[not_use_shortcut]
        num_loops = np.max(np.ceil(np.log2(num_positives[not_use_shortcut])).astype(np.int32))
        for i in range(num_loops):
            mid_index = (left_index + right_index) // 2
            right_criteria = self._total_negatives[mid_index] > neg_item_choice
            left_criteria = np.logical_not(right_criteria)
            right_index[right_criteria] = mid_index[right_criteria]
            left_index[left_criteria] = mid_index[left_criteria]
        assert np.all(right_index - left_index <= 1)
        output[not_use_shortcut] = self._sorted_train_pos_items[right_index] - (self._total_negatives[right_index] - neg_item_choice)
        assert np.all(output >= 0)
        return output

def get_constructor(name):
    if False:
        i = 10
        return i + 15
    if name == 'bisection':
        return BisectionDataConstructor
    if name == 'materialized':
        return MaterializedDataConstructor
    raise ValueError('Unrecognized constructor: {}'.format(name))