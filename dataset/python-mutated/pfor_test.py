from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.parallel_for import pfor
from tensorflow.python.platform import test

class PForTest(test.TestCase):

    def test_rank_known(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            x = array_ops.placeholder(dtypes.float32, [None, None])
            rank = pfor._rank(x)
            self.assertIsInstance(rank, int)
            self.assertEqual(rank, 2)

    def test_rank_unknown(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            x = array_ops.placeholder(dtypes.float32)
            rank = pfor._rank(x)
            self.assertIsInstance(rank, tensor.Tensor)

    def test_size_known(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            x = array_ops.placeholder(dtypes.float32, [3, 5])
            size = pfor._size(x)
            self.assertIsInstance(size, int)
            self.assertEqual(size, 3 * 5)

    def test_size_unknown(self):
        if False:
            return 10
        with ops.Graph().as_default():
            x = array_ops.placeholder(dtypes.float32, [3, None])
            size = pfor._size(x, dtypes.int32)
            self.assertIsInstance(size, tensor.Tensor)
            self.assertEqual(size.dtype, dtypes.int32)
            size = pfor._size(x, dtypes.int64)
            self.assertIsInstance(size, tensor.Tensor)
            self.assertEqual(size.dtype, dtypes.int64)

    def test_expand_dims_static(self):
        if False:
            for i in range(10):
                print('nop')
        x = random_ops.random_uniform([3, 5])
        axis = 1
        num_axes = 2
        expected = array_ops.reshape(x, [3, 1, 1, 5])
        actual = pfor._expand_dims(x, axis, num_axes)
        self.assertAllEqual(expected, actual)

    def test_expand_dims_dynamic(self):
        if False:
            while True:
                i = 10
        x = random_ops.random_uniform([3, 5])
        axis = 1
        num_axes = constant_op.constant([2])
        expected = array_ops.reshape(x, [3, 1, 1, 5])
        actual = pfor._expand_dims(x, axis, num_axes)
        self.assertAllEqual(expected, actual)
if __name__ == '__main__':
    test.main()