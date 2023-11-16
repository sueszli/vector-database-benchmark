"""Tests for wrapping an eager op in a call op at runtime."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables

@test_util.with_eager_op_as_function(only_as_function=True)
class RunEagerOpAsFunctionXlaTest(xla_test.XLATestCase):

    def testVarCreateReadDestroy(self):
        if False:
            i = 10
            return i + 15
        with self.test_scope():
            var = variables.Variable(1.0)
            value = var.read_value()
            self.assertAllEqual(value, 1.0)

    def testReadVariableInFunction(self):
        if False:
            i = 10
            return i + 15
        with self.test_scope():
            v = resource_variable_ops.ResourceVariable(1.0)

            @def_function.function
            def f():
                if False:
                    return 10
                return v.read_value()
            var = f()
            self.assertEqual(1.0, var.numpy())
if __name__ == '__main__':
    test.main()