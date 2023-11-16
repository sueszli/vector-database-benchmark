"""Tests for TPU Embeddings mid level API on TPU."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_v2_correctness_base_test

class TPUEmbeddingCorrectnessTest(tpu_embedding_v2_correctness_base_test.TPUEmbeddingCorrectnessBaseTest):

    @parameterized.parameters([True, False])
    def test_dense_lookup(self, is_high_dimensional):
        if False:
            return 10
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        if is_high_dimensional:
            dataset = self._create_high_dimensional_dense_dataset(strategy)
        else:
            dataset = self._create_dense_dataset(strategy)
        dist = strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False))
        dist_iter = iter(dist)

        @def_function.function
        def test_fn():
            if False:
                return 10

            def step():
                if False:
                    return 10
                return mid_level_api.dequeue()
            mid_level_api.enqueue(next(dist_iter), training=False)
            return strategy.run(step)
        shard_out_val = test_fn()
        shard0 = (self._unpack(strategy, shard_out_val[0]), self._unpack(strategy, shard_out_val[1]), self._unpack(strategy, shard_out_val[2]))
        numpy_videos = np.reshape(self.embedding_values, (8, 4))
        numpy_users = np.reshape(self.embedding_values, (16, 2))
        repeat_batch_num = strategy.num_replicas_in_sync // 2
        golden = (numpy_videos[self.feature_watched_values[:self.data_batch_size] * repeat_batch_num], numpy_videos[self.feature_favorited_values[:self.data_batch_size] * repeat_batch_num], numpy_users[self.feature_friends_values[:self.data_batch_size] * repeat_batch_num])
        if is_high_dimensional:
            dense_size = self.data_batch_size * self.data_batch_size
            golden = (numpy_videos[self.feature_watched_values_high_dimensional[:dense_size] * repeat_batch_num].reshape(self.data_batch_size * repeat_batch_num, self.data_batch_size, -1), numpy_videos[self.feature_favorited_values_high_dimensional[:dense_size] * repeat_batch_num].reshape(self.data_batch_size * repeat_batch_num, self.data_batch_size, -1), numpy_users[self.feature_friends_values_high_dimensional[:dense_size] * repeat_batch_num].reshape(self.data_batch_size * repeat_batch_num, self.data_batch_size, -1))
        self.assertAllClose(shard0, golden)
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()