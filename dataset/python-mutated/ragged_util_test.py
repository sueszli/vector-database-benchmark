"""Tests for ragged_util."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors import InvalidArgumentError
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.platform import googletest
TENSOR_3D = [[[('%d%d%d' % (i, j, k)).encode('utf-8') for k in range(3)] for j in range(2)] for i in range(4)]
TENSOR_4D = [[[[('%d%d%d%d' % (i, j, k, l)).encode('utf-8') for l in range(5)] for k in range(3)] for j in range(2)] for i in range(4)]

@test_util.run_all_in_graph_and_eager_modes
class RaggedUtilTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters([dict(data=['a', 'b', 'c'], repeats=[3, 0, 2], axis=0, expected=[b'a', b'a', b'a', b'c', b'c']), dict(data=[[1, 2], [3, 4]], repeats=[2, 3], axis=0, expected=[[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]), dict(data=[[1, 2], [3, 4]], repeats=[2, 3], axis=1, expected=[[1, 1, 2, 2, 2], [3, 3, 4, 4, 4]]), dict(data=['a', 'b', 'c'], repeats=2, axis=0, expected=[b'a', b'a', b'b', b'b', b'c', b'c']), dict(data=[[1, 2], [3, 4]], repeats=2, axis=0, expected=[[1, 2], [1, 2], [3, 4], [3, 4]]), dict(data=[[1, 2], [3, 4]], repeats=2, axis=1, expected=[[1, 1, 2, 2], [3, 3, 4, 4]]), dict(data=3, repeats=4, axis=0, expected=[3, 3, 3, 3]), dict(data=[3], repeats=4, axis=0, expected=[3, 3, 3, 3]), dict(data=3, repeats=[4], axis=0, expected=[3, 3, 3, 3]), dict(data=[3], repeats=[4], axis=0, expected=[3, 3, 3, 3]), dict(data=[], repeats=[], axis=0, expected=[])])
    def testRepeat(self, data, repeats, expected, axis=None):
        if False:
            print('Hello World!')
        result = ragged_util.repeat(data, repeats, axis)
        self.assertAllEqual(result, expected)

    @parameterized.parameters([dict(mode=mode, **args) for mode in ['constant', 'dynamic', 'unknown_shape'] for args in [dict(data=3, repeats=4, axis=0), dict(data=[3], repeats=4, axis=0), dict(data=3, repeats=[4], axis=0), dict(data=[3], repeats=[4], axis=0), dict(data=[], repeats=5, axis=0), dict(data=[1, 2, 3], repeats=5, axis=0), dict(data=[1, 2, 3], repeats=[3, 0, 2], axis=0), dict(data=[1, 2, 3], repeats=[3, 0, 2], axis=-1), dict(data=[b'a', b'b', b'c'], repeats=[3, 0, 2], axis=0), dict(data=[[1, 2, 3], [4, 5, 6]], repeats=3, axis=0), dict(data=[[1, 2, 3], [4, 5, 6]], repeats=3, axis=1), dict(data=[[1, 2, 3], [4, 5, 6]], repeats=[3, 5], axis=0), dict(data=[[1, 2, 3], [4, 5, 6]], repeats=[3, 5, 7], axis=1), dict(data=TENSOR_3D, repeats=2, axis=0), dict(data=TENSOR_3D, repeats=2, axis=1), dict(data=TENSOR_3D, repeats=2, axis=2), dict(data=TENSOR_3D, repeats=[2, 0, 4, 1], axis=0), dict(data=TENSOR_3D, repeats=[3, 2], axis=1), dict(data=TENSOR_3D, repeats=[1, 3, 1], axis=2), dict(data=TENSOR_4D, repeats=2, axis=0), dict(data=TENSOR_4D, repeats=2, axis=1), dict(data=TENSOR_4D, repeats=2, axis=2), dict(data=TENSOR_4D, repeats=2, axis=3), dict(data=TENSOR_4D, repeats=[2, 0, 4, 1], axis=0), dict(data=TENSOR_4D, repeats=[3, 2], axis=1), dict(data=TENSOR_4D, repeats=[1, 3, 1], axis=2), dict(data=TENSOR_4D, repeats=[1, 3, 0, 0, 2], axis=3)]])
    def testValuesMatchesNumpy(self, mode, data, repeats, axis):
        if False:
            i = 10
            return i + 15
        if axis < 0 and mode == 'unknown_shape':
            return
        expected = np.repeat(data, repeats, axis)
        if mode == 'constant':
            data = constant_op.constant(data)
            repeats = constant_op.constant(repeats)
        elif mode == 'dynamic':
            data = constant_op.constant(data)
            repeats = constant_op.constant(repeats)
            data = array_ops.placeholder_with_default(data, data.shape)
            repeats = array_ops.placeholder_with_default(repeats, repeats.shape)
        elif mode == 'unknown_shape':
            data = array_ops.placeholder_with_default(data, None)
            repeats = array_ops.placeholder_with_default(repeats, None)
        result = ragged_util.repeat(data, repeats, axis)
        self.assertAllEqual(result, expected)

    @parameterized.parameters([dict(descr='axis >= rank(data)', mode='dynamic', data=[1, 2, 3], repeats=[3, 0, 2], axis=1, error='axis=1 out of bounds: expected -1<=axis<1'), dict(descr='axis < -rank(data)', mode='dynamic', data=[1, 2, 3], repeats=[3, 0, 2], axis=-2, error='axis=-2 out of bounds: expected -1<=axis<1'), dict(descr='len(repeats) != data.shape[axis]', mode='dynamic', data=[[1, 2, 3], [4, 5, 6]], repeats=[2, 3], axis=1, error='Dimensions 3 and 2 are not compatible'), dict(descr='rank(repeats) > 1', mode='dynamic', data=[[1, 2, 3], [4, 5, 6]], repeats=[[3], [5]], axis=1, error='Shape \\(2, 1\\) must have rank at most 1'), dict(descr='non-integer axis', mode='constant', data=[1, 2, 3], repeats=2, axis='foo', exception=TypeError, error='`axis` must be an int')])
    def testError(self, descr, mode, data, repeats, axis, exception=ValueError, error=None):
        if False:
            return 10
        with self.assertRaises(exception):
            np.repeat(data, repeats, axis)
        if mode == 'constant':
            data = constant_op.constant(data)
            repeats = constant_op.constant(repeats)
        elif mode == 'dynamic':
            data = constant_op.constant(data)
            repeats = constant_op.constant(repeats)
            data = array_ops.placeholder_with_default(data, data.shape)
            repeats = array_ops.placeholder_with_default(repeats, repeats.shape)
        elif mode == 'unknown_shape':
            data = array_ops.placeholder_with_default(data, None)
            repeats = array_ops.placeholder_with_default(repeats, None)
        with self.assertRaisesRegex(exception, error):
            ragged_util.repeat(data, repeats, axis)

    @parameterized.parameters([dict(params=[1, 2, 3], splits=[-1, -3], repeats=2, exception=InvalidArgumentError), dict(params=[1, 2, 3], splits=[1, 2], repeats=0.5, exception=TypeError)])
    def testInputCheck(self, params, splits, repeats, exception):
        if False:
            i = 10
            return i + 15
        params = constant_op.constant(params)
        splits = constant_op.constant(splits)
        repeats = constant_op.constant(repeats)
        with self.assertRaises(exception):
            ragged_util.repeat_ranges(params, splits, repeats)
if __name__ == '__main__':
    googletest.main()