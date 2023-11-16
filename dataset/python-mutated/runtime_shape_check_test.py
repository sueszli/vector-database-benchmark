"""Tests for shape checks at runtime in XLA:GPU."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class RuntimeShapeCheckTest(xla_test.XLATestCase):

    def testUniqueDifferentSizes(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that we correctly check for shape mismatches at runtime.'
        if 'tpu' in self.device.lower():
            self.skipTest('We do not check shapes on TPU')
        with ops.device(f'device:{self.device}:0'):

            @def_function.function(jit_compile=True)
            def f(x, y):
                if False:
                    print('Hello World!')
                return array_ops.unique(x).y + array_ops.unique(y).y
            f(constant_op.constant([3.1, 3.2]), constant_op.constant([3.3, 3.2]))
            with self.assertRaisesRegex(errors.InternalError, 'different size'):
                f(constant_op.constant([3.1, 3.2]), constant_op.constant([3.1, 3.2, 3.3]))

    def testWhereOpDifferentSizes(self):
        if False:
            for i in range(10):
                print('nop')
        'Test shape mismatches with multiple dimensions.'
        if 'tpu' in self.device.lower():
            self.skipTest('We do not check shapes on TPU')
        with ops.device(f'device:{self.device}:0'):

            @def_function.function(jit_compile=True)
            def f(x, y):
                if False:
                    while True:
                        i = 10
                return array_ops.where(x) + array_ops.where(y)
            f(constant_op.constant([[3.1, 3.2, 0], [3.1, 3.2, 0]]), constant_op.constant([[3.3, 3.2, 0, 0, 0], [3.3, 3.2, 0, 0, 0]]))
            with self.assertRaisesRegex(errors.InternalError, 'different size'):
                f(constant_op.constant([[3.1, 3.2, 0], [3.1, 3.2, 0]]), constant_op.constant([[3.3, 3.2, 0, 0, 0], [3.3, 3.2, 3.3, 0, 0]]))
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()