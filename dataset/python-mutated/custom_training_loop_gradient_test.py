"""Tests for custom training loops."""
from absl.testing import parameterized
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

def get_dataset_from_tensor_slices(inp_array):
    if False:
        i = 10
        return i + 15
    dataset = dataset_ops.DatasetV2.from_tensor_slices(inp_array)
    if not tf2.enabled():
        dataset = dataset_ops.Dataset.from_tensor_slices(inp_array)
    return dataset

class AssertFlattenedMixin(object):
    """Mixin for specialized asserts."""

    def assert_equal_flattened(self, expected_results, actual_results):
        if False:
            return 10
        'Asserts that flattened results are equal.\n\n    Due to the number of replicas in the strategy, the output may have a\n    different structure and needs to be flattened for comparison.\n\n    Args:\n      expected_results: The results expected as a result of a computation.\n      actual_results: The actual results of a computation.\n    '
        self.assertEqual(len(expected_results), len(actual_results))
        for (i, expected_result) in enumerate(expected_results):
            final_result = []
            actual_result = actual_results[i]
            for val in actual_result:
                final_result.extend(val.numpy())
            self.assertAllEqual(expected_result, final_result)

class GradientTapeTest(test.TestCase, parameterized.TestCase, AssertFlattenedMixin):

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testStepInFunctionGradient(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0, 8.0]).batch(2)

        @def_function.function
        def train_step(x):
            if False:
                i = 10
                return i + 15

            def computation(x):
                if False:
                    print('Hello World!')
                return math_ops.square(x)
            with backprop.GradientTape() as tape:
                tape.watch(x)
                y = computation(x)
            grads = tape.gradient(y, x)
            return grads
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        results = []
        for x in dist_dataset:
            output = distribution.experimental_local_results(distribution.run(train_step, args=(x,)))
            results.append(output)
        self.assert_equal_flattened([[10.0, 12.0], [14.0, 16.0]], results)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testRunInFunctionGradient(self, distribution):
        if False:
            while True:
                i = 10
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0, 8.0]).batch(2)

        @def_function.function
        def run(x):
            if False:
                i = 10
                return i + 15

            def train_step(x):
                if False:
                    return 10

                def computation(x):
                    if False:
                        for i in range(10):
                            print('nop')
                    return math_ops.square(x)
                with backprop.GradientTape() as tape:
                    tape.watch(x)
                    y = computation(x)
                grads = tape.gradient(y, x)
                return grads
            return distribution.experimental_local_results(distribution.run(train_step, args=(x,)))
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        results = []
        for x in dist_dataset:
            output = run(x)
            results.append(output)
        self.assert_equal_flattened([[10.0, 12.0], [14.0, 16.0]], results)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager'], model_in_tf_function=[True, False]))
    def testNestedFunction(self, distribution, model_in_tf_function):
        if False:
            return 10

        def model(x):
            if False:
                while True:
                    i = 10
            return x * x
        if model_in_tf_function:
            model = def_function.function(model)
        with distribution.scope():
            x = variables.Variable(1.0)

            @def_function.function
            def train_step():
                if False:
                    return 10

                def replica_step():
                    if False:
                        print('Hello World!')
                    with backprop.GradientTape() as tape:
                        y = model(x)
                    return tape.gradient(y, x)
                return distribution.run(replica_step)
            grads = distribution.experimental_local_results(train_step())
            self.assertLen(grads, distribution.num_replicas_in_sync)
            self.assertTrue(all((g is not None for g in grads)))
if __name__ == '__main__':
    test.main()