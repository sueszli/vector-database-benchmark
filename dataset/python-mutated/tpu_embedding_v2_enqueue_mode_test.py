"""Tests for TPU Embeddings mid level API on TPU."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_base_test
from tensorflow.python.util import nest

class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

    @parameterized.parameters([True, False])
    def test_enqueue_with_outside_compilation(self, use_mlir):
        if False:
            print('Hello World!')
        if use_mlir:
            config.enable_mlir_bridge()
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        mid_level_api.build([TensorShape((self.batch_size, 2)), TensorShape((self.batch_size, 2)), TensorShape((self.batch_size, 3))])
        dataset = self._create_sparse_dataset(strategy)
        dataset_iter = iter(strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))

        @def_function.function
        def enqueue_with_outside_compilation(data):
            if False:
                return 10

            def get_activations(features):
                if False:
                    while True:
                        i = 10
                mid_level_api.enqueue(features, training=False)
                return mid_level_api.dequeue()
            return strategy.run(get_activations, args=(data,))

        @def_function.function
        def enqueue_without_outside_compilation(data):
            if False:
                return 10

            def get_activations():
                if False:
                    while True:
                        i = 10
                return mid_level_api.dequeue()
            mid_level_api.enqueue(data, training=False)
            return strategy.run(get_activations)
        features = next(dataset_iter)
        activations_oc = enqueue_with_outside_compilation(features)
        activations = enqueue_without_outside_compilation(features)
        activations_oc0 = self._get_replica_numpy(activations_oc, strategy, 0)
        activations0 = self._get_replica_numpy(activations, strategy, 0)
        self.assertAllClose(activations_oc0, activations0)

    @parameterized.parameters(True, False)
    def test_enqueue_with_outside_compilation_in_control_flow(self, use_mlir):
        if False:
            i = 10
            return i + 15
        self.skip_if_oss()
        if use_mlir:
            config.enable_mlir_bridge()
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        dataset = self._create_sparse_dataset(strategy)
        dataset_iter = iter(strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))

        @def_function.function
        def enqueue_fn(features):
            if False:
                return 10
            mid_level_api.enqueue(features, training=False)

        @def_function.function
        def enqueue_with_outside_compilation():
            if False:
                while True:
                    i = 10

            def get_activations(features):
                if False:
                    for i in range(10):
                        print('nop')
                enqueue_fn(features)
                return mid_level_api.dequeue()
            return strategy.run(get_activations, args=(next(dataset_iter),))
        with self.assertRaisesRegex(RuntimeError, 'does not match graph which contains TPUReplicateContext'):
            enqueue_with_outside_compilation()

    def test_enqueue_with_outside_compilation_non_direct_input(self):
        if False:
            print('Hello World!')
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        mid_level_api.build([TensorShape((self.batch_size, 2)), TensorShape((self.batch_size, 2)), TensorShape((self.batch_size, 3))])
        dataset = self._create_sparse_dataset(strategy)
        dataset_iter = iter(strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))

        @def_function.function
        def enqueue_with_outside_compilation():
            if False:
                print('Hello World!')

            def get_activations(features):
                if False:
                    return 10
                features = (features[0] * 2, features[1] * 2, features[2] * 2)
                mid_level_api.enqueue(features, training=False)
                return mid_level_api.dequeue()
            return strategy.run(get_activations, args=(next(dataset_iter),))
        with self.assertRaisesRegex(ValueError, 'which does not have the `_tpu_input_identity` attr'):
            enqueue_with_outside_compilation()

    def test_enqueue_with_outside_compilation_auto_mode(self):
        if False:
            i = 10
            return i + 15
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        mid_level_api.build([TensorShape((self.batch_size, 2)), TensorShape((self.batch_size, 2)), TensorShape((self.batch_size, 3))])
        dataset = self._create_sparse_dataset(strategy)
        dataset_iter = iter(strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))

        @def_function.function
        def enqueue_with_no_gradient_apply(data):
            if False:
                i = 10
                return i + 15

            def get_activations(features):
                if False:
                    print('Hello World!')
                mid_level_api.enqueue(features, name='call1')
                return mid_level_api.dequeue(name='call1')
            return strategy.run(get_activations, args=(data,))

        @def_function.function
        def enqueue_with_gradient_apply(data):
            if False:
                return 10

            def get_activations(features):
                if False:
                    print('Hello World!')
                mid_level_api.enqueue(features, name='call2')
                activations = mid_level_api.dequeue(name='call2')
                gradients = nest.map_structure(array_ops.ones_like, activations)
                mid_level_api.apply_gradients(gradients, name='call2')
                return activations
            return strategy.run(get_activations, args=(data,))
        data = next(dataset_iter)
        before_gradient_apply = enqueue_with_gradient_apply(data)
        after_gradient_apply = enqueue_with_no_gradient_apply(data)
        before_gradient_apply0 = self._get_replica_numpy(before_gradient_apply, strategy, 0)
        after_gradient_apply0 = self._get_replica_numpy(after_gradient_apply, strategy, 0)
        num_replicas = strategy.num_replicas_in_sync
        update = ([[0.3 * num_replicas], [0.3 * num_replicas * 2]], [[0.3 * num_replicas * 2], [0.3 * num_replicas]], [[0.1 * num_replicas], [0.1 / 3 * num_replicas]])
        golden = tuple([before - np.array(up) for (before, up) in zip(before_gradient_apply0, update)])
        self.assertAllClose(golden, after_gradient_apply0)
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()