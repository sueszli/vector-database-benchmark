"""Tests for integer division by zero."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class ZeroDivisionTest(test.TestCase):

    def testZeros(self):
        if False:
            i = 10
            return i + 15
        with test_util.use_gpu():
            for dtype in (dtypes.uint8, dtypes.int16, dtypes.int32, dtypes.int64):
                zero = constant_op.constant(0, dtype=dtype)
                one = constant_op.constant(1, dtype=dtype)
                bads = [lambda x, y: x // y]
                if dtype in (dtypes.int32, dtypes.int64):
                    bads.append(lambda x, y: x % y)
                for bad in bads:
                    try:
                        result = self.evaluate(bad(one, zero))
                    except (errors.OpError, errors.InvalidArgumentError) as e:
                        self.assertIn('Integer division by zero', str(e))
                    else:
                        self.assertTrue(test.is_gpu_available())
                        self.assertIn(result, (-1, 1, 2, 255, 4294967295))
if __name__ == '__main__':
    test.main()