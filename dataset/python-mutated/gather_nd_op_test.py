"""Tests for tensorflow.ops.tf.gather_nd."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class GatherNdTest(xla_test.XLATestCase):

    def _runGather(self, params, indices):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            paramsp = array_ops.placeholder(params.dtype)
            indicesp = array_ops.placeholder(indices.dtype)
            with self.test_scope():
                gather_nd_t = array_ops.gather_nd(paramsp, indicesp)
            feed_dict = {paramsp: params, indicesp: indices}
            return gather_nd_t.eval(feed_dict=feed_dict)

    def testSimpleDtype(self):
        if False:
            return 10
        for dtype in self.numeric_types:
            self.assertAllEqual(np.array([7, 7, 8], dtype=dtype), self._runGather(np.array([8, 1, 2, 3, 7, 5], dtype=dtype), np.array([[4], [4], [0]], np.int32)))

    @test_util.disable_mlir_bridge('Error handling')
    def testEmptyIndicesAndParamsOKButJustEmptyParamsFails(self):
        if False:
            while True:
                i = 10
        with self.session():
            params = np.ones((3, 3), dtype=np.float32)
            indices_empty = np.empty((0, 2), dtype=np.int32)
            gather_nd_ok_val = self._runGather(params, indices_empty)
            self.assertAllClose(np.empty((0,), dtype=np.float32), gather_nd_ok_val)
            indices_empty = np.empty((0, 1), dtype=np.int32)
            gather_nd_ok_val = self._runGather(params, indices_empty)
            self.assertAllClose(np.empty((0, 3), dtype=np.float32), gather_nd_ok_val)
            params_empty = np.empty((0, 3), dtype=np.float32)
            indices_empty = np.empty((0, 2), dtype=np.int32)
            gather_nd_ok_val = self._runGather(params_empty, indices_empty)
            self.assertAllClose(np.empty((0,), dtype=np.float32), gather_nd_ok_val)
            params_empty = np.empty((0, 3), dtype=np.float32)
            indices_nonempty = np.zeros((1, 2), dtype=np.int32)
            with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, 'Gather dimension 0 is of size zero'):
                self._runGather(params_empty, indices_nonempty)

    def testIndexScalar(self):
        if False:
            return 10
        params = np.array([[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
        indices = np.array([4, 1], dtype=np.int32)
        gather_nd_val = self._runGather(params, indices)
        self.assertAllEqual(np.array(7), gather_nd_val)

    def testParamsRankLargerThanIndexIndexScalarSlices(self):
        if False:
            i = 10
            return i + 15
        params = np.array([[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
        indices = np.array([4], dtype=np.int32)
        gather_nd_val = self._runGather(params, indices)
        self.assertAllEqual(np.array([-7, 7]), gather_nd_val)

    def testParamsRankLargerThanIndexSlices(self):
        if False:
            i = 10
            return i + 15
        params = np.array([[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
        indices = np.array([[4], [4], [0]], np.int32)
        gather_nd_val = self._runGather(params, indices)
        self.assertAllEqual(np.array([[-7, 7], [-7, 7], [-8, 8]]), gather_nd_val)

    def testHigherRankParamsLargerThanIndexSlices(self):
        if False:
            return 10
        params = np.array([[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]], dtype=np.float32).T
        indices = np.array([[4], [4], [0]], np.int32)
        gather_nd_val = self._runGather(params, indices)
        self.assertAllEqual(params[[4, 4, 0]], gather_nd_val)

    def testEmptyIndicesLastRankMeansCopyEntireTensor(self):
        if False:
            while True:
                i = 10
        params = np.array([[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]], dtype=np.float32).T
        indices = np.array([[], []], dtype=np.int32)
        gather_nd_val = self._runGather(params, indices)
        self.assertAllEqual(np.vstack((params[np.newaxis, :], params[np.newaxis, :])), gather_nd_val)

    def testHigherRankParamsAndIndicesLargerThanIndexSlices(self):
        if False:
            print('Hello World!')
        params = np.array([[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]], dtype=np.float32).T
        indices = np.array([[[3], [2], [1]], [[4], [4], [0]]], np.int32)
        gather_nd_val = self._runGather(params, indices)
        self.assertAllEqual(params[[3, 2, 1, 4, 4, 0]].reshape(2, 3, 2, 2), gather_nd_val)

    def testHigherRankParams(self):
        if False:
            for i in range(10):
                print('nop')
        shape = (10, 20, 5, 1, 17)
        params = np.random.rand(*shape).astype(np.float32)
        indices = np.vstack([np.random.randint(0, s, size=2000, dtype=np.int32) for s in shape]).T
        gather_nd_val = self._runGather(params, indices)
        expected = params[tuple(indices.T)]
        self.assertAllEqual(expected, gather_nd_val)

    def testHigherRankParamsAndIndices(self):
        if False:
            return 10
        shape = (10, 20, 5, 1, 17)
        params = np.random.rand(*shape).astype(np.float32)
        indices = np.vstack([np.random.randint(0, s, size=2000, dtype=np.int32) for s in shape]).T
        indices_reshaped = indices.reshape([10, 10, 20, 5])
        gather_nd_val = self._runGather(params, indices_reshaped)
        expected = params[tuple(indices.T)]
        self.assertAllEqual(expected.reshape([10, 10, 20]), gather_nd_val)
if __name__ == '__main__':
    test.main()