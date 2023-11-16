"""Tests for bfloat16 helper."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.tpu import bfloat16
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

class BFloat16ScopeTest(test.TestCase):

    def testDefaultScopeName(self):
        if False:
            return 10
        'Test if name for the variable scope is propagated correctly.'
        with bfloat16.bfloat16_scope() as bf:
            self.assertEqual(bf.name, '')

    def testCustomScopeName(self):
        if False:
            while True:
                i = 10
        'Test if custom name for the variable scope is propagated correctly.'
        name = 'bfloat16'
        with bfloat16.bfloat16_scope('bfloat16') as bf:
            self.assertEqual(bf.name, name)

    def testVariableName(self):
        if False:
            return 10
        'Test if custom name for the variable scope is propagated correctly.'
        g = ops.Graph()
        with g.as_default():
            a = variables.Variable(2.2, name='var_a')
            b = variables.Variable(3.3, name='var_b')
            d = variables.Variable(4.4, name='var_b')
            with g.name_scope('scope1'):
                with bfloat16.bfloat16_scope('bf16'):
                    a = math_ops.cast(a, dtypes.bfloat16)
                    b = math_ops.cast(b, dtypes.bfloat16)
                    c = math_ops.add(a, b, name='addition')
                with bfloat16.bfloat16_scope():
                    d = math_ops.cast(d, dtypes.bfloat16)
                    math_ops.add(c, d, name='addition')
        g_ops = g.get_operations()
        ops_name = []
        for op in g_ops:
            ops_name.append(str(op.name))
        self.assertIn('scope1/bf16/addition', ops_name)
        self.assertIn('scope1/bf16/Cast', ops_name)
        self.assertIn('scope1/addition', ops_name)
        self.assertIn('scope1/Cast', ops_name)

    @test_util.run_deprecated_v1
    def testRequestedDType(self):
        if False:
            while True:
                i = 10
        'Test if requested dtype is honored in the getter.\n    '
        with bfloat16.bfloat16_scope() as scope:
            v1 = variable_scope.get_variable('v1', [])
            self.assertEqual(v1.dtype.base_dtype, dtypes.float32)
            v2 = variable_scope.get_variable('v2', [], dtype=dtypes.bfloat16)
            self.assertEqual(v2.dtype.base_dtype, dtypes.bfloat16)
            self.assertEqual([dtypes.float32, dtypes.float32], [v.dtype.base_dtype for v in scope.global_variables()])
if __name__ == '__main__':
    test.main()