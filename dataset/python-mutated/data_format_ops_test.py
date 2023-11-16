"""Tests for the DataFormatVecPermute operator."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

class XlaDataFormatDimMapTest(xla_test.XLATestCase):

    def _test(self, input_data, src_format, dst_format, expected):
        if False:
            for i in range(10):
                print('nop')
        for dtype in {np.int32, np.int64}:
            x = np.array(input_data, dtype=dtype)
            with self.session() as session:
                with self.test_scope():
                    placeholder = array_ops.placeholder(dtypes.as_dtype(x.dtype), x.shape)
                    param = {placeholder: x}
                    output = nn_ops.data_format_dim_map(placeholder, src_format=src_format, dst_format=dst_format)
                result = session.run(output, param)
            self.assertAllEqual(result, expected)

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self._test(0, 'NHWC', 'NCHW', 0)
        self._test(1, 'NHWC', 'NCHW', 2)
        self._test(2, 'NHWC', 'NCHW', 3)
        self._test(3, 'NHWC', 'NCHW', 1)
        self._test(-1, 'NHWC', 'NCHW', 1)
        self._test(-2, 'NHWC', 'NCHW', 3)
        self._test(-3, 'NHWC', 'NCHW', 2)
        self._test(-4, 'NHWC', 'NCHW', 0)
        self._test([1, 3], 'NHWC', 'NCHW', [2, 1])
        self._test([1, 3, -2], 'NHWC', 'NCHW', [2, 1, 3])
        self._test([1, -3, -2], 'NHWC', 'NCHW', [2, 2, 3])
        self._test([[1, -3], [1, -1]], 'NHWC', 'NCHW', [[2, 2], [2, 1]])
        self._test([1, -3, -2], 'NHWC', 'NCHW', [2, 2, 3])
        self._test([-4, -3, -2, -1, 0, 1, 2, 3], 'NHWC', 'HWNC', [2, 0, 1, 3, 2, 0, 1, 3])
        self._test([-4, -3, -2, -1, 0, 1, 2, 3], 'NHWC', 'WHCN', [3, 1, 0, 2, 3, 1, 0, 2])
        self._test([-4, -3, -2, -1, 0, 1, 2, 3], 'qwer', 'rewq', [3, 2, 1, 0, 3, 2, 1, 0])
        self._test(0, 'NDHWC', 'NCDHW', 0)
        self._test(1, 'NDHWC', 'NCDHW', 2)
        self._test(2, 'NDHWC', 'NCDHW', 3)
        self._test(3, 'NDHWC', 'NCDHW', 4)
        self._test(4, 'NDHWC', 'NCDHW', 1)
        self._test([1, 4], 'NDHWC', 'NCDHW', [2, 1])
        self._test([1, 4, -2], 'NDHWC', 'NCDHW', [2, 1, 4])
        self._test([1, -3, -2], 'NDHWC', 'NCDHW', [2, 3, 4])
        self._test([[1, -4], [1, -1]], 'NDHWC', 'NCDHW', [[2, 2], [2, 1]])
        self._test([1, -3, -2], 'NDHWC', 'NCDHW', [2, 3, 4])
        self._test([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], 'NDHWC', 'DHWNC', [3, 0, 1, 2, 4, 3, 0, 1, 2, 4])
        self._test([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], 'NDHWC', 'WHDCN', [4, 2, 1, 0, 3, 4, 2, 1, 0, 3])

class XlaPermuteOpTest(xla_test.XLATestCase):

    def _runPermuteAndCompare(self, x, src_format, dst_format, expected):
        if False:
            while True:
                i = 10
        with self.session() as session:
            with self.test_scope():
                placeholder = array_ops.placeholder(dtypes.as_dtype(x.dtype), x.shape)
                param = {placeholder: x}
                output = nn_ops.data_format_vec_permute(placeholder, src_format=src_format, dst_format=dst_format)
            result = session.run(output, param)
        self.assertAllEqual(result, expected)

    def testNHWCToNCHW(self):
        if False:
            while True:
                i = 10
        for dtype in {np.int32, np.int64}:
            x = np.array([7, 4, 9, 3], dtype=dtype)
            self._runPermuteAndCompare(x, 'NHWC', 'NCHW', [7, 3, 4, 9])

    def testNHWCToNCHW_Size2(self):
        if False:
            while True:
                i = 10
        for dtype in {np.int32, np.int64}:
            x = np.array([4, 9], dtype=dtype)
            self._runPermuteAndCompare(x, 'NHWC', 'NCHW', [4, 9])

    def testNCHWToNHWC(self):
        if False:
            return 10
        for dtype in {np.int32, np.int64}:
            x = np.array([7, 4, 9, 3], dtype=dtype)
            self._runPermuteAndCompare(x, 'NCHW', 'NHWC', [7, 9, 3, 4])

    def testNCHWToNHWC_Size2(self):
        if False:
            while True:
                i = 10
        for dtype in {np.int32, np.int64}:
            x = np.array([9, 3], dtype=dtype)
            self._runPermuteAndCompare(x, 'NCHW', 'NHWC', [9, 3])

    def testNHWCToHWNC(self):
        if False:
            print('Hello World!')
        for dtype in {np.int32, np.int64}:
            x = np.array([7, 4, 9, 3], dtype=dtype)
            self._runPermuteAndCompare(x, 'NHWC', 'HWNC', [4, 9, 7, 3])

    def testHWNCToNHWC(self):
        if False:
            while True:
                i = 10
        for dtype in {np.int32, np.int64}:
            x = np.array([7, 4, 9, 3], dtype=dtype)
            self._runPermuteAndCompare(x, 'HWNC', 'NHWC', [9, 7, 4, 3])

    def testNHWCToNCHW2D(self):
        if False:
            return 10
        for dtype in {np.int32, np.int64}:
            x = np.array([[7, 4], [9, 3], [4, 5], [5, 1]], dtype=dtype)
            self._runPermuteAndCompare(x, 'NHWC', 'NCHW', [[7, 4], [5, 1], [9, 3], [4, 5]])

    def testNHWCToHWNC2D(self):
        if False:
            i = 10
            return i + 15
        for dtype in {np.int32, np.int64}:
            x = np.array([[7, 4], [9, 3], [4, 5], [5, 1]], dtype=dtype)
            self._runPermuteAndCompare(x, 'NHWC', 'HWNC', [[9, 3], [4, 5], [7, 4], [5, 1]])

    def testHWNCToNHWC2D(self):
        if False:
            return 10
        for dtype in {np.int32, np.int64}:
            x = np.array([[7, 4], [9, 3], [4, 5], [5, 1]], dtype=dtype)
            self._runPermuteAndCompare(x, 'HWNC', 'NHWC', [[4, 5], [7, 4], [9, 3], [5, 1]])

    def testNCHWToNHWC2D(self):
        if False:
            return 10
        for dtype in {np.int32, np.int64}:
            x = np.array([[7, 4], [9, 3], [4, 5], [5, 1]], dtype=dtype)
            self._runPermuteAndCompare(x, 'NCHW', 'NHWC', [[7, 4], [4, 5], [5, 1], [9, 3]])
if __name__ == '__main__':
    test.main()