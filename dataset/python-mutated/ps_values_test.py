"""Tests for the distributed values library."""
import os
from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_lib

def async_checkpoint_test_helper(test_case, x):
    if False:
        return 10
    test_case.evaluate(x.assign(123.0))
    checkpoint = trackable_utils.Checkpoint(x=x)
    ckpt_options = checkpoint_options.CheckpointOptions(experimental_enable_async_checkpoint=True)
    prefix = os.path.join(test_case.get_temp_dir(), 'ckpt')
    save_path = checkpoint.save(prefix, options=ckpt_options)
    test_case.evaluate(x.assign(234.0))
    test_case.assertNotAllClose(123.0, x.read_value())
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    test_case.assertEqual(test_case.evaluate(x), 123.0)
    test_case.evaluate(x.assign(345.0))
    save_path = checkpoint.save(prefix, options=ckpt_options)
    test_case.evaluate(x.assign(456.0))
    test_case.assertNotAllClose(345.0, x.read_value())
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    test_case.assertEqual(test_case.evaluate(x), 345.0)

@combinations.generate(combinations.combine(distribution=[strategy_combinations.central_storage_strategy_with_two_gpus], mode=['graph', 'eager']))
class AggregatingVariableTest(test.TestCase, parameterized.TestCase):

    def testAssignOutOfScope(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        with distribution.scope():
            aggregating = variables_lib.Variable(1.0)
        self.assertIsInstance(aggregating, ps_values.AggregatingVariable)
        self.evaluate(aggregating.assign(3.0))
        self.assertEqual(self.evaluate(aggregating.read_value()), 3.0)
        self.assertEqual(self.evaluate(aggregating._v.read_value()), 3.0)

    def testAssignAdd(self, distribution):
        if False:
            i = 10
            return i + 15
        with distribution.scope():
            v = variable_v1.VariableV1(1, aggregation=variables_lib.VariableAggregation.MEAN)
        self.evaluate(variables_lib.global_variables_initializer())

        @def_function.function
        def assign():
            if False:
                return 10
            return v.assign_add(2)
        per_replica_results = self.evaluate(distribution.experimental_local_results(distribution.run(assign)))
        self.assertAllEqual([3], per_replica_results)

    def testAsyncCheckpointAggregatingVariable(self, distribution):
        if False:
            return 10
        with self.test_session():
            with distribution.scope():
                x = variables_lib.Variable(1.0)
            self.assertIsInstance(x, ps_values.AggregatingVariable)
            self.evaluate(x.initializer)
            async_checkpoint_test_helper(self, x)

    def testAsyncCheckpointCachingVariable(self, distribution):
        if False:
            return 10
        del distribution
        with self.test_session():
            v = variables_lib.Variable(1.0)
            x = ps_values.CachingVariable(v)
            self.assertIsInstance(x, ps_values.CachingVariable)
            self.evaluate(x.initializer)
            async_checkpoint_test_helper(self, x)
if __name__ == '__main__':
    test.main()