"""Tests for TPU Embeddings mid level API on TPU."""
from absl.testing import parameterized
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_base_test

class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

    @parameterized.parameters([True, False])
    def test_sequence_feature(self, is_sparse):
        if False:
            return 10
        seq_length = 3
        for feature in self.feature_config:
            feature.max_sequence_length = seq_length
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        if is_sparse:
            dataset = self._create_sparse_dataset(strategy)
        else:
            dataset = self._create_ragged_dataset(strategy)
        feature_iter = iter(strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))

        @def_function.function
        def test_fn():
            if False:
                print('Hello World!')

            def step():
                if False:
                    return 10
                return mid_level_api.dequeue()
            mid_level_api.enqueue(next(feature_iter), training=False)
            return strategy.run(step)
        output = test_fn()
        self.assertEqual(self._get_replica_numpy(output[0], strategy, 0).shape, (2, 3, 4))
        self.assertEqual(self._get_replica_numpy(output[1], strategy, 0).shape, (2, 3, 4))
        self.assertEqual(self._get_replica_numpy(output[2], strategy, 0).shape, (2, 3, 2))

    @parameterized.parameters([True, False])
    def test_sequence_feature_with_build(self, is_updated_shape):
        if False:
            i = 10
            return i + 15
        seq_length = 3
        for feature in self.feature_config:
            feature.max_sequence_length = seq_length
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        dataset = self._create_sparse_dataset(strategy)
        feature_iter = iter(strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))
        if is_updated_shape:
            mid_level_api.build([TensorShape([self.batch_size, seq_length, 2]), TensorShape([self.batch_size, seq_length, 2]), TensorShape([self.batch_size, seq_length, 3])])
        else:
            mid_level_api.build([TensorShape([self.batch_size, 2]), TensorShape([self.batch_size, 2]), TensorShape([self.batch_size, 3])])

        @def_function.function
        def test_fn():
            if False:
                for i in range(10):
                    print('nop')

            def step():
                if False:
                    for i in range(10):
                        print('nop')
                return mid_level_api.dequeue()
            mid_level_api.enqueue(next(feature_iter), training=False)
            return strategy.run(step)
        output = test_fn()
        self.assertEqual(self._get_replica_numpy(output[0], strategy, 0).shape, (2, 3, 4))
        self.assertEqual(self._get_replica_numpy(output[1], strategy, 0).shape, (2, 3, 4))
        self.assertEqual(self._get_replica_numpy(output[2], strategy, 0).shape, (2, 3, 2))
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()