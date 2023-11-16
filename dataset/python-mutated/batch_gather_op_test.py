"""Tests for tensorflow.ops.tf.gather."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
_TEST_TYPES = (dtypes.int64, dtypes.float32, dtypes.complex64, dtypes.complex128)

class GatherTest(test.TestCase, parameterized.TestCase):

    def _buildParams(self, data, dtype):
        if False:
            print('Hello World!')
        data = data.astype(dtype.as_numpy_dtype)
        if dtype.is_complex:
            return data + 10j * data
        return data

    @parameterized.parameters(dtypes.int32, dtypes.int64)
    def testSimpleGather(self, indices_dtype):
        if False:
            while True:
                i = 10
        data = np.array([0, 1, 2, 3, 7, 5, 8, 9, 10, 11, 15, 13])
        indices = [3, 4]
        with self.session():
            for dtype in _TEST_TYPES:
                params_np = self._buildParams(data, dtype)
                params = constant_op.constant(params_np)
                indices_tf = constant_op.constant(indices, dtype=indices_dtype)
                gather_t = array_ops.batch_gather(params, indices_tf)
                expected_result = np.array([3, 7])
                np_val = self._buildParams(expected_result, dtype)
                gather_val = self.evaluate(gather_t)
                self.assertAllEqual(np_val, gather_val)
                self.assertEqual(np_val.shape, gather_t.get_shape())

    @parameterized.parameters(dtypes.int32, dtypes.int64)
    def test2DArray(self, indices_dtype):
        if False:
            i = 10
            return i + 15
        data = np.array([[0, 1, 2, 3, 7, 5], [8, 9, 10, 11, 15, 13]])
        indices = [[3], [4]]
        with self.session():
            for dtype in _TEST_TYPES:
                params_np = self._buildParams(data, dtype)
                params = constant_op.constant(params_np)
                indices_tf = constant_op.constant(indices, dtype=indices_dtype)
                gather_t = array_ops.batch_gather(params, indices_tf)
                expected_result = np.array([[3], [15]])
                np_val = self._buildParams(expected_result, dtype)
                gather_val = self.evaluate(gather_t)
                self.assertAllEqual(np_val, gather_val)
                self.assertEqual(np_val.shape, gather_t.get_shape())

    def testHigherRank(self):
        if False:
            while True:
                i = 10
        data = np.array([[[0, 1, 2], [3, 7, 5]], [[8, 9, 10], [11, 15, 13]]])
        indices = [[[2, 0], [1, 2]], [[2, 0], [0, 1]]]
        with self.session():
            for dtype in _TEST_TYPES:
                params_np = self._buildParams(data, dtype)
                params = constant_op.constant(params_np)
                indices_tf = constant_op.constant(indices)
                gather_t = array_ops.batch_gather(params, indices_tf)
                gather_val = self.evaluate(gather_t)
                expected_result = np.array([[[2, 0], [7, 5]], [[10, 8], [11, 15]]])
                np_val = self._buildParams(expected_result, dtype)
                self.assertAllEqual(np_val, gather_val)
                self.assertEqual(np_val.shape, gather_t.get_shape())

    def testString(self):
        if False:
            print('Hello World!')
        params = np.array([[b'asdf', b'zxcv'], [b'qwer', b'uiop']])
        with self.cached_session():
            indices_tf = constant_op.constant([1])
            self.assertAllEqual([[b'qwer', b'uiop']], self.evaluate(array_ops.batch_gather(params, indices_tf)))

    def testUnknownIndices(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            params = constant_op.constant([[0, 1, 2]])
            indices = array_ops.placeholder(dtypes.int32, shape=[None, None])
            gather_t = array_ops.batch_gather(params, indices)
            self.assertEqual([1, None], gather_t.get_shape().as_list())

    @test_util.disable_xla('Cannot force cpu placement for xla_gpu test')
    def testBadIndicesCPU(self):
        if False:
            print('Hello World!')
        with ops.device_v2('cpu:0'):
            params = [[0, 1, 2], [3, 4, 5]]
            with self.assertRaisesOpError('indices\\[0\\] = 7 is not in \\[0, 2\\)'):
                self.evaluate(array_ops.batch_gather(params, [7]))

    def testEmptySlices(self):
        if False:
            while True:
                i = 10
        with self.session():
            for dtype in _TEST_TYPES:
                for itype in (np.int32, np.int64):
                    params = np.zeros((7, 0, 0), dtype=dtype.as_numpy_dtype)
                    indices = np.array([3, 4], dtype=itype)
                    self.assertAllEqual(self.evaluate(array_ops.batch_gather(params, indices)), np.zeros((2, 0, 0)))
if __name__ == '__main__':
    test.main()