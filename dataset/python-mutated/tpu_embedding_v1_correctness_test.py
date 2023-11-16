"""Tests for TPU Embeddings mid level API on TPU."""
from absl.testing import parameterized
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v1
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test

class TPUEmbeddingV0CorrectnessTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

    def _get_strategy(self):
        if False:
            return 10
        if hasattr(self, 'strategy'):
            return self.strategy
        return super(TPUEmbeddingV0CorrectnessTest, self)._get_strategy()

    def _create_mid_level(self, optimizer=None):
        if False:
            return 10
        if optimizer is None:
            optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
        return tpu_embedding_v1.TPUEmbeddingV0(feature_config=self.feature_config, optimizer=optimizer)

    def _create_strategy_and_mid_level(self, optimizer_name):
        if False:
            i = 10
            return i + 15
        strategy = self._get_strategy()
        with strategy.scope():
            if optimizer_name == 'sgd':
                embedding_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
            elif optimizer_name == 'adagrad':
                embedding_optimizer = tpu_embedding_v2_utils.Adagrad(learning_rate=0.1)
            elif optimizer_name == 'adam':
                embedding_optimizer = tpu_embedding_v2_utils.Adam(learning_rate=0.1)
            elif optimizer_name == 'ftrl':
                embedding_optimizer = tpu_embedding_v2_utils.FTRL(learning_rate=0.1)
            else:
                raise ValueError('optimizer is not recognized: ', optimizer_name)
            mid_level_api = self._create_mid_level(optimizer=embedding_optimizer)
        return (strategy, mid_level_api)

    @parameterized.parameters(True, False)
    def test_enqueue_with_weights(self, ragged):
        if False:
            return 10
        (strategy, mid_level_api) = self._create_strategy_and_mid_level('sgd')
        weight = 0.5
        if ragged:
            dataset = self._create_ragged_dataset(strategy, include_weights=True, weight=weight)
        else:
            dataset = self._create_sparse_dataset(strategy, include_weights=True, weight=weight)
        dataset_iter = iter(strategy.experimental_distribute_dataset(dataset, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))

        @def_function.function
        def embedding_lookup(features, weights):
            if False:
                print('Hello World!')

            def step(features, weights):
                if False:
                    i = 10
                    return i + 15
                return mid_level_api(features, weights)
            return strategy.run(step, args=(features, weights))
        (features, weights) = next(dataset_iter)
        weights = (weights[0], None, weights[2])
        no_weights_activations = embedding_lookup(features, weights=None)
        weights_activations = embedding_lookup(features, weights=weights)
        no_weights0 = (self._unpack(strategy, no_weights_activations[0]), self._unpack(strategy, no_weights_activations[1]), self._unpack(strategy, no_weights_activations[2]))
        weights0 = (self._unpack(strategy, weights_activations[0]), self._unpack(strategy, weights_activations[1]), self._unpack(strategy, weights_activations[2]))
        weight = (0.5, 1.0, 1.0)
        golden = tuple([no_weight * w for (no_weight, w) in zip(no_weights0, weight)])
        self.assertAllClose(golden, weights0)
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()