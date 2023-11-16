"""Tests for TPU Embeddings mid level API on TPU."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test
from tensorflow.python.util import nest

class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

    def test_pass_none_to_apply_gradients(self):
        if False:
            while True:
                i = 10
        self.skip_if_oss()
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        mid_level_api.build([TensorShape((self.batch_size, 2)), TensorShape((self.batch_size, 2)), TensorShape((self.batch_size, 3))])
        dataset = self._create_sparse_dataset(strategy)
        data = next(iter(strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False))))

        @def_function.function
        def embedding_and_set_gradients(data):
            if False:
                while True:
                    i = 10
            mid_level_api.enqueue(data)

            def tpu_fn():
                if False:
                    while True:
                        i = 10
                results = mid_level_api.dequeue()
                mid_level_api.apply_gradients((None, None, array_ops.ones_like(results[2])))
                return results
            return strategy.run(tpu_fn)

        @def_function.function
        def embedding_only(data):
            if False:
                for i in range(10):
                    print('nop')
            mid_level_api.enqueue(data, training=False)

            def tpu_fn():
                if False:
                    while True:
                        i = 10
                return mid_level_api.dequeue()
            return strategy.run(tpu_fn)
        first = self._get_replica_numpy(embedding_and_set_gradients(data), strategy, 0)
        second = self._get_replica_numpy(embedding_only(data), strategy, 0)
        num_replicas = strategy.num_replicas_in_sync
        update = ([[0.0]], [[0.0]], [[0.1 * num_replicas], [0.1 / 3 * num_replicas]])
        golden = tuple([feature - np.array(up) for (feature, up) in zip(first, update)])
        self.assertAllClose(golden, second)

    def test_enqueue_sparse_and_ragged(self):
        if False:
            i = 10
            return i + 15
        self.skip_if_oss()
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        sparse = self._create_sparse_dataset(strategy)
        ragged = self._create_ragged_dataset(strategy)
        sparse_iter = iter(strategy.experimental_distribute_dataset(sparse, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))
        ragged_iter = iter(strategy.experimental_distribute_dataset(ragged, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))

        @def_function.function
        def test_fn():
            if False:
                while True:
                    i = 10

            def step():
                if False:
                    print('Hello World!')
                return mid_level_api.dequeue()
            sparse_features = next(sparse_iter)
            ragged_features = next(ragged_iter)
            features = (sparse_features[0], ragged_features[1], sparse_features[2])
            mid_level_api.enqueue(features, training=False)
            return strategy.run(step)
        test_fn()

    def test_enqueue_per_device(self):
        if False:
            return 10
        self.skip_if_oss()
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        sparse = self._create_sparse_dataset(strategy)
        sparse_iter = iter(strategy.experimental_distribute_dataset(sparse, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))

        @def_function.function
        def test_fn():
            if False:
                while True:
                    i = 10

            def get_activations(dense_value):
                if False:
                    for i in range(10):
                        print('nop')
                return (mid_level_api.dequeue(), dense_value)
            sparse_features = next(sparse_iter)
            mid_level_api.enqueue(sparse_features, training=False)
            (activations, dense_value1) = strategy.run(get_activations, args=(0.0,))

            def enqueue_fn(ctx):
                if False:
                    print('Hello World!')
                core_id = ctx.replica_id_in_sync_group
                device = strategy.extended.worker_devices[core_id]
                sparse_features_local = nest.map_structure(lambda x: strategy.experimental_local_results(x)[core_id], sparse_features)
                mid_level_api.enqueue(sparse_features_local, training=False, device=device)
                return 0.0
            data = strategy.experimental_distribute_values_from_function(enqueue_fn)
            (per_device_activations, dense_value2) = strategy.run(get_activations, args=(data,))
            return (activations, per_device_activations, dense_value1, dense_value2)
        (activations, per_device_activations, _, _) = test_fn()
        activations0 = self._get_replica_numpy(activations, strategy, 0)
        per_device_activations0 = self._get_replica_numpy(per_device_activations, strategy, 0)
        self.assertAllClose(activations0, per_device_activations0)
        test_fn()

    @parameterized.parameters(True, False)
    def test_enqueue_with_weights(self, ragged):
        if False:
            i = 10
            return i + 15
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        weight = 0.5
        if ragged:
            dataset = self._create_ragged_dataset(strategy, include_weights=True, weight=weight)
        else:
            dataset = self._create_sparse_dataset(strategy, include_weights=True, weight=weight)
            mid_level_api.build([TensorShape((self.batch_size, 2)), TensorShape((self.batch_size, 2)), TensorShape((self.batch_size, 3))])
        dataset_iter = iter(strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))

        @def_function.function
        def enqueue_and_get(features, weights):
            if False:
                while True:
                    i = 10

            def get_activations():
                if False:
                    i = 10
                    return i + 15
                return mid_level_api.dequeue()
            mid_level_api.enqueue(features, weights=weights, training=False)
            return strategy.run(get_activations)
        (features, weights) = next(dataset_iter)
        weights = (weights[0], None, weights[2])
        no_weights_activations = enqueue_and_get(features, weights=None)
        weights_activations = enqueue_and_get(features, weights=weights)
        no_weights0 = self._get_replica_numpy(no_weights_activations, strategy, 0)
        weights0 = self._get_replica_numpy(weights_activations, strategy, 0)
        weight = (0.5, 1.0, 1.0)
        golden = tuple([no_weight * w for (no_weight, w) in zip(no_weights0, weight)])
        self.assertAllClose(golden, weights0)

    def test_same_config_different_instantiations(self):
        if False:
            for i in range(10):
                print('nop')
        self.skip_if_oss()
        num_tables = 30
        table_dim = np.random.randint(1, 128, size=[num_tables])
        table_vocab_size = np.random.randint(100, 1000, size=[num_tables])
        table_names = ['table{}'.format(i) for i in range(num_tables)]
        table_data = list(zip(table_dim, table_vocab_size, table_names))
        strategy = self._get_strategy()

        def tpu_embedding_config():
            if False:
                print('Hello World!')
            feature_configs = []
            for (dim, vocab, name) in table_data:
                feature_configs.append(tpu_embedding_v2_utils.FeatureConfig(table=tpu_embedding_v2_utils.TableConfig(vocabulary_size=int(vocab), dim=int(dim), initializer=init_ops_v2.Zeros(), name=name)))
            optimizer = tpu_embedding_v2_utils.Adagrad(learning_rate=0.1)
            with strategy.scope():
                mid_level_api = tpu_embedding_v2.TPUEmbedding(feature_config=feature_configs, optimizer=optimizer)
            mid_level_api._output_shapes = [TensorShape(128)] * len(feature_configs)
            return mid_level_api._create_config_proto()
        self.assertProtoEquals(tpu_embedding_config(), tpu_embedding_config())

    @parameterized.parameters([True, False])
    def test_missing_feature(self, is_sparse):
        if False:
            for i in range(10):
                print('nop')
        strategy = self._get_strategy()
        with strategy.scope():
            optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
            mid_level_api = tpu_embedding_v2.TPUEmbedding(feature_config=tpu_embedding_v2_utils.FeatureConfig(table=self.table_video, name='watched'), optimizer=optimizer)
        if is_sparse:
            features = sparse_tensor.SparseTensor(indices=self.feature_watched_indices[:-1], values=self.feature_watched_values[:-1], dense_shape=[self.data_batch_size, 2])
        else:
            features = ragged_tensor.RaggedTensor.from_row_lengths(row_lengths=[1, 2, 2, 0], values=self.feature_watched_values[:-1])
        dataset = dataset_ops.DatasetV2.from_tensors(features)
        dataset = dataset.unbatch().repeat().batch(self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)
        dataset_iter = iter(strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))

        @def_function.function
        def test_fn():
            if False:
                i = 10
                return i + 15

            def get_activations():
                if False:
                    for i in range(10):
                        print('nop')
                return mid_level_api.dequeue()
            mid_level_api.enqueue(next(dataset_iter), training=False)
            return strategy.run(get_activations)
        test_fn()
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()