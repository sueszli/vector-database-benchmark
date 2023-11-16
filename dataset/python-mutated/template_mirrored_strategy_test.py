"""Tests for make_template used with MirroredStrategy."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class TemplateMirroredStrategyTest(test.TestCase):

    @test_util.disable_tfrt('Strategy not supported yet.')
    def test_merge_call(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            if not test.is_gpu_available():
                self.skipTest('No GPU available')

            def fn():
                if False:
                    print('Hello World!')
                var1 = variable_scope.get_variable('var1', shape=[], initializer=init_ops.constant_initializer(21.0))
                distribute_lib.get_replica_context().merge_call(lambda _: ())
                var2 = variable_scope.get_variable('var2', shape=[], initializer=init_ops.constant_initializer(2.0))
                return var1 * var2
            temp = template.make_template('my_template', fn)
            strategy = mirrored_strategy.MirroredStrategy(['/cpu:0', '/gpu:0'])
            out = strategy.experimental_local_results(strategy.run(temp))
            self.evaluate(variables.global_variables_initializer())
            self.assertAllEqual([42.0, 42.0], self.evaluate(out))
if __name__ == '__main__':
    test.main()