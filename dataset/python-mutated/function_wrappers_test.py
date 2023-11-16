"""Tests for function_wrappers module."""
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class FunctionWrappersTest(test.TestCase):

    def test_name_scope(self):
        if False:
            i = 10
            return i + 15
        if context.executing_eagerly():
            self.skipTest('Tensor names are disabled in eager')
        with function_wrappers.FunctionScope('test_name', None, converter.ConversionOptions(optional_features=converter.Feature.NAME_SCOPES)):
            t = constant_op.constant(1)
        self.assertIn('test_name', t.name)

    def test_auto_control_deps(self):
        if False:
            i = 10
            return i + 15
        v = variables.Variable(1)
        with function_wrappers.FunctionScope('_', None, converter.ConversionOptions(optional_features=converter.Feature.AUTO_CONTROL_DEPS)) as scope:
            v.assign(2)
            op = scope.ret(constant_op.constant(1), True)
        self.evaluate(op)
        self.assertEqual(self.evaluate(v.read_value()), 2)

    def test_all_disabled(self):
        if False:
            return 10
        with function_wrappers.FunctionScope(None, None, converter.STANDARD_OPTIONS):
            t = constant_op.constant(1)
        self.assertEqual(self.evaluate(t), 1)
if __name__ == '__main__':
    test.main()