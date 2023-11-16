"""Tests for array operations."""
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import weak_tensor_ops
from tensorflow.python.platform import test

class ArrayOpTest(test.TestCase):

    def testReshapeShapeInference(self):
        if False:
            while True:
                i = 10
        x = weak_tensor.WeakTensor(random_ops.random_normal([4, 10, 10]))
        x.shape.assert_is_compatible_with([4, None, 10])
        a = array_ops.reshape(x, array_ops.shape(x))
        a.shape.assert_is_compatible_with([4, None, 10])
        b = array_ops.reshape(x, math_ops.cast(array_ops.shape(x), dtypes.int64))
        b.shape.assert_is_compatible_with([4, None, 10])
        c = array_ops.reshape(x, math_ops.cast(math_ops.cast(array_ops.shape(x), dtypes.float32), dtypes.int32))
        c.shape.assert_is_compatible_with([None, None, None])
        self.assertIsInstance(c, weak_tensor.WeakTensor)

    def testSlicedPartialShapeInference(self):
        if False:
            for i in range(10):
                print('nop')

        @def_function.function(autograph=False)
        def g(x):
            if False:
                return 10
            return array_ops.zeros([array_ops.shape(x)[0]])
        conc = g.get_concrete_function(tensor_spec.TensorSpec([10, None]))
        self.assertAllEqual(conc.output_shapes.as_list(), [10])

    def testIdentityOnSlicedPartialShapeInference(self):
        if False:
            for i in range(10):
                print('nop')

        @def_function.function(autograph=False)
        def g(x):
            if False:
                for i in range(10):
                    print('nop')
            return array_ops.zeros([array_ops.identity(array_ops.shape(x)[0])])
        conc = g.get_concrete_function(tensor_spec.TensorSpec([10, None]))
        self.assertAllEqual(conc.output_shapes.as_list(), [10])
if __name__ == '__main__':
    ops.set_dtype_conversion_mode('all')
    test.main()