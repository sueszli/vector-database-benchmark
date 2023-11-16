"""Tests for IdentityNOp."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class IdentityNOpTest(test.TestCase):

    def testInt32String_6(self):
        if False:
            i = 10
            return i + 15
        (value0, value1) = self.evaluate(array_ops.identity_n([[1, 2, 3, 4, 5, 6], [b'a', b'b', b'C', b'd', b'E', b'f', b'g']]))
        self.assertAllEqual(np.array([1, 2, 3, 4, 5, 6]), value0)
        self.assertAllEqual(np.array([b'a', b'b', b'C', b'd', b'E', b'f', b'g']), value1)

    def testInt32_shapes(self):
        if False:
            for i in range(10):
                print('nop')
        inp0 = constant_op.constant([10, 20, 30, 40, 50, 60], shape=[2, 3])
        inp1 = constant_op.constant([11, 21, 31, 41, 51, 61], shape=[3, 2])
        inp2 = constant_op.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], shape=[5, 3])
        (value0, value1, value2) = self.evaluate(array_ops.identity_n([inp0, inp1, inp2]))
        self.assertAllEqual(np.array([[10, 20, 30], [40, 50, 60]]), value0)
        self.assertAllEqual(np.array([[11, 21], [31, 41], [51, 61]]), value1)
        self.assertAllEqual(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]), value2)

    def testString(self):
        if False:
            for i in range(10):
                print('nop')
        source = [b'A', b'b', b'C', b'd', b'E', b'f']
        [value] = self.evaluate(array_ops.identity_n([source]))
        self.assertAllEqual(source, value)

    def testIdentityShape(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            shape = [2, 3]
            array_2x3 = [[1, 2, 3], [6, 5, 4]]
            tensor = constant_op.constant(array_2x3)
            self.assertEqual(shape, tensor.get_shape())
            self.assertEqual(shape, array_ops.identity_n([tensor])[0].get_shape())
            self.assertEqual(shape, array_ops.identity_n([array_2x3])[0].get_shape())
if __name__ == '__main__':
    test.main()