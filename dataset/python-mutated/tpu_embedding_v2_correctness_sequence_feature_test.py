"""Tests for TPU Embeddings mid level API on TPU."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_v2_correctness_base_test
from tensorflow.python.util import nest

class TPUEmbeddingCorrectnessTest(tpu_embedding_v2_correctness_base_test.TPUEmbeddingCorrectnessBaseTest):

    @parameterized.parameters([True, False])
    def test_sequence_embeddings(self, sparse):
        if False:
            print('Hello World!')
        feature_config = (tpu_embedding_v2_utils.FeatureConfig(table=self.table_video, name='watched', max_sequence_length=2), tpu_embedding_v2_utils.FeatureConfig(table=self.table_video, name='favorited', max_sequence_length=2), tpu_embedding_v2_utils.FeatureConfig(table=self.table_user, name='friends', max_sequence_length=3))
        optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
        strategy = self._get_strategy()
        num_replicas = strategy.num_replicas_in_sync
        with strategy.scope():
            mid_level = tpu_embedding_v2.TPUEmbedding(feature_config=feature_config, optimizer=optimizer)
        mid_level.build(self.batch_size)
        if sparse:
            dataset = self._create_sparse_dataset(strategy)
        else:
            dataset = self._create_ragged_dataset(strategy)
        data = next(iter(strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False))))

        @def_function.function
        def embedding_and_set_gradients(data):
            if False:
                for i in range(10):
                    print('nop')

            def tpu_fn():
                if False:
                    for i in range(10):
                        print('nop')
                activations = mid_level.dequeue()
                mid_level.apply_gradients(nest.map_structure(array_ops.ones_like, activations))
                return activations
            mid_level.enqueue(data)
            return strategy.run(tpu_fn)

        @def_function.function
        def embedding_only(data):
            if False:
                return 10

            def tpu_fn():
                if False:
                    return 10
                return mid_level.dequeue()
            mid_level.enqueue(data, training=False)
            return strategy.run(tpu_fn)
        before_update = self._get_replica_numpy(embedding_and_set_gradients(data), strategy, 0)
        after_update = self._get_replica_numpy(embedding_only(data), strategy, 0)
        masks = (np.array([[[1], [0]], [[1], [1]]]), np.array([[[1], [1]], [[1], [0]]]), np.array([[[1], [0], [0]], [[1], [1], [1]]]))
        per_row_update = (0.3 * num_replicas, 0.3 * num_replicas, 0.1 * num_replicas)
        golden = tuple([before - update * mask for (before, update, mask) in zip(before_update, per_row_update, masks)])
        self.assertAllClose(golden, after_update)
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()