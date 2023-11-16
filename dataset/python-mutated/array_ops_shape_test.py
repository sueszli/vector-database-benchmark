"""Tests for shape op int64 output."""
from tensorflow.core.config import flags
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class ArrayOpShapeSizeTest(test.TestCase):

    def testShapeInt64Flag(self):
        if False:
            return 10
        self.assertTrue(flags.config().tf_shape_default_int64.value())
        s1 = array_ops.shape_v2(array_ops.zeros([1, 2]))
        self.assertEqual(s1.dtype, dtypes.int64)

    def testShapeInt64FlagTf1(self):
        if False:
            while True:
                i = 10
        self.assertTrue(flags.config().tf_shape_default_int64.value())
        s1 = array_ops.shape(array_ops.zeros([1, 2]))
        self.assertEqual(s1.dtype, dtypes.int64)

    def testSizeInt64Flag(self):
        if False:
            while True:
                i = 10
        self.assertTrue(flags.config().tf_shape_default_int64.value())
        s1 = array_ops.size_v2(array_ops.zeros([1, 2]))
        self.assertEqual(s1.dtype, dtypes.int64)

    def testSizeInt64FlagTf1(self):
        if False:
            print('Hello World!')
        self.assertTrue(flags.config().tf_shape_default_int64.value())
        s1 = array_ops.size(array_ops.zeros([1, 2]))
        self.assertEqual(s1.dtype, dtypes.int64)
if __name__ == '__main__':
    test.main()