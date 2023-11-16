"""Data loader and input processing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v2 as tf
from typing import Text, Optional
from official.modeling.hyperparams import params_dict
from official.vision.detection.dataloader import factory
from official.vision.detection.dataloader import mode_keys as ModeKeys

class InputFn(object):
    """Input function for tf.Estimator."""

    def __init__(self, file_pattern: Text, params: params_dict.ParamsDict, mode: Text, batch_size: int, num_examples: Optional[int]=-1):
        if False:
            return 10
        'Initialize.\n\n    Args:\n      file_pattern: the file pattern for the data example (TFRecords).\n      params: the parameter object for constructing example parser and model.\n      mode: ModeKeys.TRAIN or ModeKeys.Eval\n      batch_size: the data batch size.\n      num_examples: If positive, only takes this number of examples and raise\n        tf.errors.OutOfRangeError after that. If non-positive, it will be\n        ignored.\n    '
        assert file_pattern is not None
        assert mode is not None
        assert batch_size is not None
        self._file_pattern = file_pattern
        self._mode = mode
        self._is_training = mode == ModeKeys.TRAIN
        self._batch_size = batch_size
        self._num_examples = num_examples
        self._parser_fn = factory.parser_generator(params, mode)
        self._dataset_fn = tf.data.TFRecordDataset
        self._input_sharding = not self._is_training
        try:
            if self._is_training:
                self._input_sharding = params.train.input_sharding
            else:
                self._input_sharding = params.eval.input_sharding
        except KeyError:
            pass

    def __call__(self, ctx=None, batch_size: int=None):
        if False:
            i = 10
            return i + 15
        'Provides tf.data.Dataset object.\n\n    Args:\n      ctx: context object.\n      batch_size: expected batch size input data.\n\n    Returns:\n      tf.data.Dataset object.\n    '
        if not batch_size:
            batch_size = self._batch_size
        assert batch_size is not None
        dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=self._is_training)
        if self._input_sharding and ctx and (ctx.num_input_pipelines > 1):
            dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
        if self._is_training:
            dataset = dataset.repeat()
        dataset = dataset.interleave(map_func=lambda file_name: self._dataset_fn(file_name), cycle_length=32, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()
        if self._is_training:
            dataset = dataset.shuffle(64)
        if self._num_examples > 0:
            dataset = dataset.take(self._num_examples)
        dataset = dataset.map(self._parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset