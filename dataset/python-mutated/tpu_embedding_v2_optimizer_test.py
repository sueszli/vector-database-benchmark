"""Tests for TPU Embeddings mid level API on TPU."""
import functools
from absl.testing import parameterized
import numpy as np
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test

class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

    def test_unsupported_optimizer(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(ValueError, 'is an unsupported optimizer class.'):
            with self._get_strategy().scope():
                tpu_embedding_v2.TPUEmbedding(self.feature_config, tpu_embedding.AdagradParameters(learning_rate=0.1))

    def test_variable_learning_rate(self):
        if False:
            i = 10
            return i + 15
        num_steps = 10
        num_steps_float = float(num_steps)
        starting_lr = 1.0
        ending_lr = 0.5
        strategy = self._get_strategy()
        num_replicas = strategy.num_replicas_in_sync
        with strategy.scope():
            step_counter = tf_variables.Variable(0.0, dtypes.float32)

            def lr_function():
                if False:
                    return 10
                return gen_math_ops.maximum(ending_lr, starting_lr + (ending_lr - starting_lr) * step_counter / num_steps_float)
            optimizer = tpu_embedding_v2_utils.SGD(learning_rate=lr_function)
            table_config = tpu_embedding_v2_utils.TableConfig(vocabulary_size=num_replicas, dim=4, initializer=init_ops_v2.Constant(np.zeros((num_replicas, 4))), combiner='sum', name='table')
            mid_level_api = tpu_embedding_v2.TPUEmbedding(feature_config={'feature': tpu_embedding_v2_utils.FeatureConfig(table=table_config, name='feature')}, optimizer=optimizer)
        feature = {'feature': constant_op.constant([0], shape=(1, 1), dtype=dtypes.int32)}

        def input_fn(ctx):
            if False:
                return 10
            del ctx
            return dataset_ops.DatasetV2.from_tensors(feature).repeat()
        dist = strategy.distribute_datasets_from_function(input_fn, options=distribute_lib.InputOptions(experimental_fetch_to_device=False))
        dist_iter = iter(dist)

        @def_function.function
        def test_fn():
            if False:
                print('Hello World!')

            def step():
                if False:
                    return 10
                with backprop.GradientTape() as tape:
                    activations = mid_level_api.dequeue()
                    tape.watch(activations)
                    result = math_ops.reduce_sum(activations['feature'])
                    loss = result / num_replicas
                grads = tape.gradient(loss, activations)
                mid_level_api.apply_gradients(grads)
                return activations['feature']
            mid_level_api.enqueue(next(dist_iter), training=True)
            return strategy.run(step)
        results = []
        for _ in range(num_steps):
            result = test_fn()
            results.append(self._unpack(strategy, result))
            step_counter.assign_add(1.0)
        learning_rates = [starting_lr - (starting_lr - ending_lr) / num_steps * j for j in range(num_steps)]
        cumsum = [sum(learning_rates[0:j]) for j in range(num_steps)]
        goldens = [[[-cumsum[i]] * table_config.dim] * num_replicas for i in range(10)]
        self.assertAllClose(results, goldens)

    @parameterized.parameters([True, False])
    def test_optimizer_with_slot_creation_fn(self, use_tpu):
        if False:
            print('Hello World!')

        def slot_creation_fn(table, slot_names, _):
            if False:
                while True:
                    i = 10
            slots = {}
            for slot in slot_names:
                slots[slot] = tf_variables.Variable(name='{}_{}'.format(table.name, slot), initial_value=functools.partial(init_ops_v2.Zeros(), shape=table.shape, dtype=dtypes.float32), trainable=False)
            return slots
        optimizer = tpu_embedding_v2_utils.Adagrad(learning_rate=0.1, slot_variable_creation_fn=slot_creation_fn)
        if use_tpu:
            strategy = self._get_strategy()
        else:
            strategy = distribute_lib.get_strategy()
        with strategy.scope():
            mid_level = tpu_embedding_v2.TPUEmbedding(feature_config=self.feature_config, optimizer=optimizer)
            mid_level.build(self.batch_size)
        video_accumulator = mid_level._variables['video']['accumulators']
        user_accumulator = mid_level._variables['user']['accumulators']
        if use_tpu:
            video_accumulator = video_accumulator.variables[0]
            user_accumulator = user_accumulator.variables[0]
        self.assertAllClose(video_accumulator.numpy(), np.zeros((self.table_video.vocabulary_size, self.table_video.dim)))
        self.assertAllClose(user_accumulator.numpy(), np.zeros((self.table_user.vocabulary_size, self.table_user.dim)))

    def test_optimizer_with_slot_creation_fn_non_partial(self):
        if False:
            print('Hello World!')

        def slot_creation_fn(table, slot_names, _):
            if False:
                while True:
                    i = 10
            slots = {}
            for slot in slot_names:
                slots[slot] = tf_variables.Variable(name='{}_{}'.format(table.name, slot), initial_value=init_ops_v2.Zeros()(shape=table.shape, dtype=dtypes.float32), trainable=False)
            return slots
        optimizer = tpu_embedding_v2_utils.Adagrad(learning_rate=0.1, slot_variable_creation_fn=slot_creation_fn)
        strategy = self._get_strategy()
        with strategy.scope():
            mid_level_api = tpu_embedding_v2.TPUEmbedding(feature_config=self.feature_config, optimizer=optimizer)
            with self.assertRaisesRegex(ValueError, 'Unable to extract initializer function'):
                mid_level_api.build(self.batch_size)
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()