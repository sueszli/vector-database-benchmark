"""Tests for TPU Embeddings mid level API on TPU."""
import numpy as np
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_base_test

class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

    def test_enqueue_dense_sparse_ragged(self):
        if False:
            for i in range(10):
                print('nop')
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        dataset = self._create_high_dimensional_dense_dataset(strategy)
        dense_iter = iter(strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))
        sparse = self._create_high_dimensional_sparse_dataset(strategy)
        sparse_iter = iter(strategy.experimental_distribute_dataset(sparse, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))
        ragged = self._create_high_dimensional_ragged_dataset(strategy)
        ragged_iter = iter(strategy.experimental_distribute_dataset(ragged, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))
        mid_level_api.build([TensorShape([self.batch_size, self.data_batch_size, 1]), TensorShape([self.batch_size, self.data_batch_size, 2]), TensorShape([self.batch_size, self.data_batch_size, 3])])

        @def_function.function
        def test_fn():
            if False:
                while True:
                    i = 10

            def step():
                if False:
                    for i in range(10):
                        print('nop')
                return mid_level_api.dequeue()
            features = (next(dense_iter)[0], next(sparse_iter)[1], next(ragged_iter)[2])
            mid_level_api.enqueue(features, training=False)
            return strategy.run(step)
        test_fn()

    def test_different_input_shapes(self):
        if False:
            for i in range(10):
                print('nop')
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        sparse = self._create_high_dimensional_sparse_dataset(strategy)
        sparse_iter = iter(strategy.experimental_distribute_dataset(sparse, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))
        dense_feature = constant_op.constant(np.zeros(3), shape=(1, 3, 1), dtype=dtypes.int32)
        dense_dataset = dataset_ops.DatasetV2.from_tensors(dense_feature).unbatch().repeat().batch(1 * strategy.num_replicas_in_sync, drop_remainder=True)
        dense_iter = iter(strategy.experimental_distribute_dataset(dense_dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))

        @def_function.function
        def test_fn():
            if False:
                return 10

            def step():
                if False:
                    i = 10
                    return i + 15
                return mid_level_api.dequeue()
            features = (next(dense_iter), next(sparse_iter)[1], next(sparse_iter)[2])
            mid_level_api.enqueue(features, training=False)
            return strategy.run(step)
        test_fn()
        self.assertEqual(mid_level_api._output_shapes, [TensorShape((1, 3)), TensorShape((self.batch_size, self.data_batch_size)), TensorShape((self.batch_size, self.data_batch_size))])

    def test_output_shapes_priority_over_feature_config_and_build(self):
        if False:
            print('Hello World!')
        (_, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        mid_level_api._output_shapes = [TensorShape((2, 4)) for _ in range(3)]
        mid_level_api.build([TensorShape((2, None, None)) for _ in range(3)])
        self.assertEqual(mid_level_api._output_shapes, [TensorShape((2, 4)) for _ in range(3)])
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()