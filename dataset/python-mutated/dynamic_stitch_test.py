"""Tests for tf.dynamic_stitch."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import googletest

class DynamicStitchTest(xla_test.XLATestCase):

    def _AssertDynamicStitchResultIs(self, indices, data, expected):
        if False:
            for i in range(10):
                print('nop')
        with self.session() as session:
            index_placeholders = [array_ops.placeholder(dtypes.as_dtype(arg.dtype)) for arg in indices]
            data_placeholders = [array_ops.placeholder(dtypes.as_dtype(arg.dtype)) for arg in data]
            with self.test_scope():
                output = data_flow_ops.dynamic_stitch(index_placeholders, data_placeholders)
            feed_dict = {}
            for (placeholder, value) in zip(index_placeholders, indices):
                feed_dict[placeholder] = value
            for (placeholder, value) in zip(data_placeholders, data):
                feed_dict[placeholder] = value
            result = session.run(output, feed_dict=feed_dict)
            self.assertAllClose(expected, result, rtol=0.001)

    def testSimpleEmpty(self):
        if False:
            return 10
        idx1 = np.array([0, 2], dtype=np.int32)
        idx2 = np.array([[1], [3]], dtype=np.int32)
        val1 = np.array([[], []], dtype=np.int32)
        val2 = np.array([[[]], [[]]], dtype=np.int32)
        self._AssertDynamicStitchResultIs([idx1, idx2], [val1, val2], expected=np.array([[], [], [], []], np.int32))

    def testEmptyIndex(self):
        if False:
            for i in range(10):
                print('nop')
        idx1 = np.array([], dtype=np.int32)
        idx2 = np.array([[], []], dtype=np.int32)
        val1 = np.ndarray(shape=(0, 9), dtype=np.int32)
        val2 = np.ndarray(shape=(2, 0, 9), dtype=np.int32)
        self._AssertDynamicStitchResultIs([idx1, idx2], [val1, val2], expected=np.ndarray(shape=(0, 9), dtype=np.int32))

    def testSimple1D(self):
        if False:
            print('Hello World!')
        val1 = np.array([0, 4, 7], dtype=np.int32)
        val2 = np.array([1, 6, 2, 3, 5], dtype=np.int32)
        val3 = np.array([0, 40, 70], dtype=np.float32)
        val4 = np.array([10, 60, 20, 30, 50], dtype=np.float32)
        expected = np.array([0, 10, 20, 30, 40, 50, 60, 70], dtype=np.float32)
        self._AssertDynamicStitchResultIs([val1, val2], [val3, val4], expected=expected)

    def testSimple2D(self):
        if False:
            return 10
        val1 = np.array([0, 4, 7], dtype=np.int32)
        val2 = np.array([1, 6], dtype=np.int32)
        val3 = np.array([2, 3, 5], dtype=np.int32)
        val4 = np.array([[0, 1], [40, 41], [70, 71]], dtype=np.float32)
        val5 = np.array([[10, 11], [60, 61]], dtype=np.float32)
        val6 = np.array([[20, 21], [30, 31], [50, 51]], dtype=np.float32)
        expected = np.array([[0, 1], [10, 11], [20, 21], [30, 31], [40, 41], [50, 51], [60, 61], [70, 71]], dtype=np.float32)
        self._AssertDynamicStitchResultIs([val1, val2, val3], [val4, val5, val6], expected=expected)
if __name__ == '__main__':
    googletest.main()