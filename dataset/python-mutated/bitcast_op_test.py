"""Tests for tf.bitcast."""
import sys
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

@test_util.with_eager_op_as_function
class BitcastTest(test.TestCase):

    def _testBitcast(self, x, datatype, shape):
        if False:
            i = 10
            return i + 15
        with test_util.use_gpu():
            tf_ans = array_ops.bitcast(x, datatype)
            out = self.evaluate(tf_ans)
            if sys.byteorder == 'little':
                buff_after = memoryview(out).tobytes()
                buff_before = memoryview(x).tobytes()
            else:
                buff_after = memoryview(out.byteswap()).tobytes()
                buff_before = memoryview(x.byteswap()).tobytes()
            self.assertEqual(buff_before, buff_after)
            self.assertEqual(tf_ans.get_shape(), shape)
            self.assertEqual(tf_ans.dtype, datatype)

    def testSmaller(self):
        if False:
            return 10
        x = np.random.rand(3, 2)
        datatype = dtypes.int8
        shape = [3, 2, 8]
        self._testBitcast(x, datatype, shape)

    def testLarger(self):
        if False:
            return 10
        x = np.arange(16, dtype=np.int8).reshape([4, 4])
        datatype = dtypes.int32
        shape = [4]
        self._testBitcast(x, datatype, shape)

    def testSameDtype(self):
        if False:
            print('Hello World!')
        x = np.random.rand(3, 4)
        shape = [3, 4]
        self._testBitcast(x, x.dtype, shape)

    def testSameSize(self):
        if False:
            return 10
        x = np.random.rand(3, 4)
        shape = [3, 4]
        self._testBitcast(x, dtypes.int64, shape)

    def testErrors(self):
        if False:
            print('Hello World!')
        x = np.zeros([1, 1], np.int8)
        datatype = dtypes.int32
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'Cannot bitcast from 6 to 3|convert from s8.* to S32'):
            array_ops.bitcast(x, datatype, None)

    def testEmpty(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.ones([], np.int32)
        datatype = dtypes.int8
        shape = [4]
        self._testBitcast(x, datatype, shape)

    def testUnknownShape(self):
        if False:
            return 10
        with ops.Graph().as_default():
            x = array_ops.placeholder(dtypes.float32)
            datatype = dtypes.int8
            array_ops.bitcast(x, datatype, None)

    @test_util.disable_tfrt('b/169901260')
    def testQuantizedType(self):
        if False:
            while True:
                i = 10
        shape = [3, 4]
        x = np.zeros(shape, np.uint16)
        datatype = dtypes.quint16
        self._testBitcast(x, datatype, shape)

    def testUnsignedType(self):
        if False:
            while True:
                i = 10
        shape = [3, 4]
        x = np.zeros(shape, np.int64)
        datatype = dtypes.uint64
        self._testBitcast(x, datatype, shape)
if __name__ == '__main__':
    test.main()