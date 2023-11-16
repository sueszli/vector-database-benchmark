"""Tests for TPU Embeddings mid level API on TPU."""
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_base_test

class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

    def test_build_incorrect_output_shapes(self):
        if False:
            print('Hello World!')
        (_, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        mid_level_api._output_shapes = [TensorShape((2, 4)) for _ in range(3)]
        with self.assertRaisesRegex(ValueError, 'Inconsistent shape founded for input feature'):
            mid_level_api.build([TensorShape([1, 1, 1]) for _ in range(3)])

    def test_enqueue_incorrect_shape_feature(self):
        if False:
            for i in range(10):
                print('nop')
        (strategy, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        sparse = self._create_high_dimensional_sparse_dataset(strategy)
        sparse_iter = iter(strategy.experimental_distribute_dataset(sparse, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))
        mid_level_api._output_shapes = [TensorShape((1, 1)) for _ in range(3)]
        mid_level_api.build([TensorShape([1, 1, 1]) for _ in range(3)])

        @def_function.function
        def test_fn():
            if False:
                return 10

            def step():
                if False:
                    return 10
                return mid_level_api.dequeue()
            mid_level_api.enqueue(next(sparse_iter), training=False)
            return strategy.run(step)
        with self.assertRaisesRegex(ValueError, 'Inconsistent shape founded for input feature'):
            test_fn()

    def test_not_fully_defined_output_shapes_in_feature_config(self):
        if False:
            for i in range(10):
                print('nop')
        (_, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        mid_level_api._output_shapes = [TensorShape(None) for _ in range(3)]
        with self.assertRaisesRegex(ValueError, 'Input Feature'):
            mid_level_api.build()

    def test_not_fully_defined_output_shapes_for_build(self):
        if False:
            print('Hello World!')
        (_, mid_level_api, _) = self._create_strategy_and_mid_level('sgd')
        with self.assertRaisesRegex(ValueError, 'Input Feature'):
            mid_level_api.build([TensorShape([1, None, None]) for _ in range(3)])
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()