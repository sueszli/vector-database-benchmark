"""Test cases for segment reduction ops."""
import functools
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.client import device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

class SegmentReductionOpsTest(xla_test.XLATestCase):
    """Test cases for segment reduction ops."""

    def _findDevice(self, device_name):
        if False:
            return 10
        devices = device_lib.list_local_devices()
        for d in devices:
            if d.device_type == device_name:
                return True
        return False

    def _segmentReduction(self, op, data, indices, num_segments):
        if False:
            print('Hello World!')
        with self.session() as sess, self.test_scope():
            d = array_ops.placeholder(data.dtype, shape=data.shape)
            if isinstance(indices, int):
                i = array_ops.placeholder(np.int32, shape=[])
            else:
                i = array_ops.placeholder(indices.dtype, shape=indices.shape)
            return sess.run(op(d, i, num_segments), {d: data, i: indices})

    def _unsortedSegmentSum(self, data, indices, num_segments):
        if False:
            print('Hello World!')
        return self._segmentReduction(math_ops.unsorted_segment_sum, data, indices, num_segments)

    def _segmentSumV2(self, data, indices, num_segments):
        if False:
            i = 10
            return i + 15
        return self._segmentReduction(math_ops.segment_sum_v2, data, indices, num_segments)

    def _segmentProdV2(self, data, indices, num_segments):
        if False:
            i = 10
            return i + 15
        return self._segmentReduction(math_ops.segment_prod_v2, data, indices, num_segments)

    def _segmentMinV2(self, data, indices, num_segments):
        if False:
            print('Hello World!')
        return self._segmentReduction(math_ops.segment_min_v2, data, indices, num_segments)

    def _segmentMaxV2(self, data, indices, num_segments):
        if False:
            i = 10
            return i + 15
        return self._segmentReduction(math_ops.segment_max_v2, data, indices, num_segments)

    def _unsortedSegmentProd(self, data, indices, num_segments):
        if False:
            while True:
                i = 10
        return self._segmentReduction(math_ops.unsorted_segment_prod, data, indices, num_segments)

    def _unsortedSegmentMin(self, data, indices, num_segments):
        if False:
            for i in range(10):
                print('nop')
        return self._segmentReduction(math_ops.unsorted_segment_min, data, indices, num_segments)

    def _unsortedSegmentMax(self, data, indices, num_segments):
        if False:
            print('Hello World!')
        return self._segmentReduction(math_ops.unsorted_segment_max, data, indices, num_segments)

    def testSegmentSum(self):
        if False:
            print('Hello World!')
        for dtype in self.numeric_types:
            self.assertAllClose(np.array([1, 0, 2, 12], dtype=dtype), self._segmentSumV2(np.array([0, 1, 2, 3, 4, 5], dtype=dtype), np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 4))

    def testSegmentProd(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.numeric_types:
            self.assertAllClose(np.array([0, 1, 2, 60], dtype=dtype), self._segmentProdV2(np.array([0, 1, 2, 3, 4, 5], dtype=dtype), np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 4))

    def testSegmentProdNumSegmentsLess(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.numeric_types:
            self.assertAllClose(np.array([0, 1, 2], dtype=dtype), self._segmentProdV2(np.array([0, 1, 2, 3, 4, 5], dtype=dtype), np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 3))

    def testSegmentProdNumSegmentsMore(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.numeric_types:
            self.assertAllClose(np.array([0, 1, 2, 60, 1], dtype=dtype), self._segmentProdV2(np.array([0, 1, 2, 3, 4, 5], dtype=dtype), np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 5))

    def testSegmentMin(self):
        if False:
            print('Hello World!')
        for dtype in self.int_types | self.float_types:
            maxval = dtypes.as_dtype(dtype).max
            if dtype == np.float64 and self._findDevice('TPU'):
                maxval = np.Inf
            self.assertAllClose(np.array([0, maxval, 2, 3], dtype=dtype), self._segmentMinV2(np.array([0, 1, 2, 3, 4, 5], dtype=dtype), np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 4))

    def testSegmentMinNumSegmentsLess(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.int_types | self.float_types:
            maxval = dtypes.as_dtype(dtype).max
            if dtype == np.float64 and self._findDevice('TPU'):
                maxval = np.Inf
            self.assertAllClose(np.array([0, maxval, 2], dtype=dtype), self._segmentMinV2(np.array([0, 1, 2, 3, 4, 5], dtype=dtype), np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 3))

    def testSegmentMinNumSegmentsMore(self):
        if False:
            return 10
        for dtype in self.int_types | self.float_types:
            maxval = dtypes.as_dtype(dtype).max
            if dtype == np.float64 and self._findDevice('TPU'):
                maxval = np.Inf
            self.assertAllClose(np.array([0, maxval, 2, 3, maxval], dtype=dtype), self._segmentMinV2(np.array([0, 1, 2, 3, 4, 5], dtype=dtype), np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 5))

    def testSegmentMax(self):
        if False:
            while True:
                i = 10
        for dtype in self.int_types | self.float_types:
            minval = dtypes.as_dtype(dtype).min
            if dtype == np.float64 and self._findDevice('TPU'):
                minval = -np.Inf
            self.assertAllClose(np.array([1, minval, 2, 5], dtype=dtype), self._segmentMaxV2(np.array([0, 1, 2, 3, 4, 5], dtype=dtype), np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 4))

    def testSegmentMaxNumSegmentsLess(self):
        if False:
            return 10
        for dtype in self.int_types | self.float_types:
            minval = dtypes.as_dtype(dtype).min
            if dtype == np.float64 and self._findDevice('TPU'):
                minval = -np.Inf
            self.assertAllClose(np.array([1, minval, 2], dtype=dtype), self._segmentMaxV2(np.array([0, 1, 2, 3, 4, 5], dtype=dtype), np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 3))

    def testSegmentMaxNumSegmentsMore(self):
        if False:
            print('Hello World!')
        for dtype in self.int_types | self.float_types:
            minval = dtypes.as_dtype(dtype).min
            if dtype == np.float64 and self._findDevice('TPU'):
                minval = -np.Inf
            self.assertAllClose(np.array([1, minval, 2, 5, minval], dtype=dtype), self._segmentMaxV2(np.array([0, 1, 2, 3, 4, 5], dtype=dtype), np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 5))

    def testUnsortedSegmentSum0DIndices1DData(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.numeric_types:
            self.assertAllClose(np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]], dtype=dtype), self._unsortedSegmentSum(np.array([0, 1, 2, 3, 4, 5], dtype=dtype), 2, 4))

    def testUnsortedSegmentSum1DIndices1DData(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.numeric_types:
            self.assertAllClose(np.array([1, 3, 2, 9], dtype=dtype), self._unsortedSegmentSum(np.array([0, 1, 2, 3, 4, 5], dtype=dtype), np.array([3, 0, 2, 1, 3, 3], dtype=np.int32), 4))

    def testUnsortedSegmentSum1DIndices1DDataNegativeIndices(self):
        if False:
            while True:
                i = 10
        for dtype in self.numeric_types:
            self.assertAllClose(np.array([6, 3, 0, 6], dtype=dtype), self._unsortedSegmentSum(np.array([0, 1, 2, 3, 4, 5, 6], dtype=dtype), np.array([3, -1, 0, 1, 0, -1, 3], dtype=np.int32), 4))

    def testUnsortedSegmentSum1DIndices2DDataDisjoint(self):
        if False:
            print('Hello World!')
        for dtype in self.numeric_types:
            data = np.array([[0, 1, 2, 3], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43], [50, 51, 52, 53]], dtype=dtype)
            indices = np.array([8, 1, 0, 3, 7], dtype=np.int32)
            num_segments = 10
            y = self._unsortedSegmentSum(data, indices, num_segments)
            self.assertAllClose(np.array([[30, 31, 32, 33], [20, 21, 22, 23], [0, 0, 0, 0], [40, 41, 42, 43], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [50, 51, 52, 53], [0, 1, 2, 3], [0, 0, 0, 0]], dtype=dtype), y)

    def testUnsortedSegmentSum1DIndices2DDataNonDisjoint(self):
        if False:
            while True:
                i = 10
        for dtype in self.numeric_types:
            data = np.array([[0, 1, 2, 3], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43], [50, 51, 52, 53]], dtype=dtype)
            indices = np.array([0, 1, 2, 0, 1], dtype=np.int32)
            num_segments = 4
            y = self._unsortedSegmentSum(data, indices, num_segments)
            self.assertAllClose(np.array([[40, 42, 44, 46], [70, 72, 74, 76], [30, 31, 32, 33], [0, 0, 0, 0]], dtype=dtype), y)

    def testUnsortedSegmentSum2DIndices3DData(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.numeric_types:
            data = np.array([[[0, 1, 2], [10, 11, 12]], [[100, 101, 102], [110, 111, 112]], [[200, 201, 202], [210, 211, 212]], [[300, 301, 302], [310, 311, 312]]], dtype=dtype)
            indices = np.array([[3, 5], [3, 1], [5, 0], [6, 2]], dtype=np.int32)
            num_segments = 8
            y = self._unsortedSegmentSum(data, indices, num_segments)
            self.assertAllClose(np.array([[210, 211, 212], [110, 111, 112], [310, 311, 312], [100, 102, 104], [0, 0, 0.0], [210, 212, 214], [300, 301, 302], [0, 0, 0]], dtype=dtype), y)

    def testUnsortedSegmentSum1DIndices3DData(self):
        if False:
            print('Hello World!')
        for dtype in self.numeric_types:
            data = np.array([[[0, 1, 2], [10, 11, 12]], [[100, 101, 102], [110, 111, 112]], [[200, 201, 202], [210, 211, 212]], [[300, 301, 302], [310, 311, 312]]], dtype=dtype)
            indices = np.array([3, 0, 2, 5], dtype=np.int32)
            num_segments = 6
            y = self._unsortedSegmentSum(data, indices, num_segments)
            self.assertAllClose(np.array([[[100, 101, 102.0], [110, 111, 112]], [[0, 0, 0], [0, 0, 0]], [[200, 201, 202], [210, 211, 212]], [[0, 1, 2.0], [10, 11, 12]], [[0, 0, 0], [0, 0, 0]], [[300, 301, 302], [310, 311, 312]]], dtype=dtype), y)

    def testUnsortedSegmentSumShapeError(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.numeric_types:
            data = np.ones((4, 8, 7), dtype=dtype)
            indices = np.ones((3, 2), dtype=np.int32)
            num_segments = 4
            self.assertRaises(ValueError, functools.partial(self._segmentReduction, math_ops.unsorted_segment_sum, data, indices, num_segments))

    def testUnsortedSegmentOps1DIndices1DDataNegativeIndices(self):
        if False:
            print('Hello World!')
        'Tests for min, max, and prod ops.\n\n    These share most of their implementation with sum, so we only test basic\n    functionality.\n    '
        for dtype in self.numeric_types:
            self.assertAllClose(np.array([8, 3, 1, 0], dtype=dtype), self._unsortedSegmentProd(np.array([0, 1, 2, 3, 4, 5, 6], dtype=dtype), np.array([3, -1, 0, 1, 0, -1, 3], dtype=np.int32), 4))
        for dtype in self.int_types | self.float_types:
            minval = dtypes.as_dtype(dtype).min
            maxval = dtypes.as_dtype(dtype).max
            self.assertAllClose(np.array([2, 3, maxval, 0], dtype=dtype), self._unsortedSegmentMin(np.array([0, 1, 2, 3, 4, 5, 6], dtype=dtype), np.array([3, -1, 0, 1, 0, -1, 3], dtype=np.int32), 4))
            self.assertAllClose(np.array([4, 3, minval, 6], dtype=dtype), self._unsortedSegmentMax(np.array([0, 1, 2, 3, 4, 5, 6], dtype=dtype), np.array([3, -1, 0, 1, 0, -1, 3], dtype=np.int32), 4))
if __name__ == '__main__':
    googletest.main()