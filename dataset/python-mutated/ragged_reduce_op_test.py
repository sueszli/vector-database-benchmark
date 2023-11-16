"""Tests for ragged_math_ops.reduce_<AGGREGATE> ops."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.platform import googletest
_MAX_INT32 = dtypes.int32.max
_MIN_INT32 = dtypes.int32.min
_NAN = np.nan

def mean(*values):
    if False:
        i = 10
        return i + 15
    return 1.0 * sum(values) / len(values)

def variance(*values):
    if False:
        while True:
            i = 10
    return np.var(values, dtype=np.float64)

def std(*values):
    if False:
        print('Hello World!')
    return np.std(values, dtype=np.float64)

@test_util.run_all_in_graph_and_eager_modes
class RaggedReduceOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters(dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=0, keepdims=False, expected=[15, 12, 4]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=-2, keepdims=False, expected=[15, 12, 4]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=1, keepdims=False, expected=[8, 6, 9, 8]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=-1, keepdims=False, expected=[8, 6, 9, 8]), dict(ragged_reduce_op=ragged_math_ops.reduce_prod, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=0, keepdims=False, expected=[54, 30, 4]), dict(ragged_reduce_op=ragged_math_ops.reduce_prod, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=1, keepdims=False, expected=[12, 5, 9, 12]), dict(ragged_reduce_op=ragged_math_ops.reduce_min, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=0, keepdims=False, expected=[1, 1, 4]), dict(ragged_reduce_op=ragged_math_ops.reduce_min, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=1, keepdims=False, expected=[1, 1, 9, 2]), dict(ragged_reduce_op=ragged_math_ops.reduce_max, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=0, keepdims=False, expected=[9, 6, 4]), dict(ragged_reduce_op=ragged_math_ops.reduce_max, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=1, keepdims=False, expected=[4, 5, 9, 6]), dict(ragged_reduce_op=ragged_math_ops.reduce_mean, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=0, keepdims=False, expected=[3.75, 4, 4]), dict(ragged_reduce_op=ragged_math_ops.reduce_variance, rt_input=[[3, 1, 4], [1, 1], [9], [2, 1]], axis=0, keepdims=False, expected=[9.6875, 0.0, 0.0]), dict(ragged_reduce_op=ragged_math_ops.reduce_std, rt_input=[[3, 1, 4], [3, 1], [2], [2, 1]], axis=0, keepdims=False, expected=[0.5, 0.0, 0.0]), dict(ragged_reduce_op=ragged_math_ops.reduce_any, rt_input=[[True, True], [True, True, False, True], [False, True]], axis=0, keepdims=False, expected=[True, True, False, True]), dict(ragged_reduce_op=ragged_math_ops.reduce_any, rt_input=[[True, True], [True, True, False, True], [False, True]], axis=1, keepdims=False, expected=[True, True, True]), dict(ragged_reduce_op=ragged_math_ops.reduce_all, rt_input=[[True, True], [True, True, False, True], [False, True]], axis=0, keepdims=False, expected=[False, True, False, True]), dict(ragged_reduce_op=ragged_math_ops.reduce_all, rt_input=[[True, True], [True, True, False, True], [False, True]], axis=1, keepdims=False, expected=[True, False, False]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=0, keepdims=True, expected=[[15, 12, 4]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=-2, keepdims=True, expected=[[15, 12, 4]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=1, keepdims=True, expected=[[8], [6], [9], [8]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=-1, keepdims=True, expected=[[8], [6], [9], [8]]), dict(ragged_reduce_op=ragged_math_ops.reduce_prod, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=0, keepdims=True, expected=[[54, 30, 4]]), dict(ragged_reduce_op=ragged_math_ops.reduce_prod, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=1, keepdims=True, expected=[[12], [5], [9], [12]]), dict(ragged_reduce_op=ragged_math_ops.reduce_min, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=0, keepdims=True, expected=[[1, 1, 4]]), dict(ragged_reduce_op=ragged_math_ops.reduce_min, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=1, keepdims=True, expected=[[1], [1], [9], [2]]), dict(ragged_reduce_op=ragged_math_ops.reduce_max, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=0, keepdims=True, expected=[[9, 6, 4]]), dict(ragged_reduce_op=ragged_math_ops.reduce_max, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=1, keepdims=True, expected=[[4], [5], [9], [6]]), dict(ragged_reduce_op=ragged_math_ops.reduce_mean, rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]], axis=0, keepdims=True, expected=[[3.75, 4, 4]]), dict(ragged_reduce_op=ragged_math_ops.reduce_variance, rt_input=[[3, 1, 4], [1, 1], [9], [2, 1]], axis=0, keepdims=True, expected=[[9.6875, 0.0, 0.0]]), dict(ragged_reduce_op=ragged_math_ops.reduce_std, rt_input=[[3, 1, 4], [3, 1], [2], [2, 1]], axis=0, keepdims=True, expected=[[0.5, 0.0, 0.0]]), dict(ragged_reduce_op=ragged_math_ops.reduce_any, rt_input=[[True, True], [True, True, False, True], [False, True]], axis=0, keepdims=True, expected=[[True, True, False, True]]), dict(ragged_reduce_op=ragged_math_ops.reduce_any, rt_input=[[True, True], [True, True, False, True], [False, True]], axis=1, keepdims=True, expected=[[True], [True], [True]]), dict(ragged_reduce_op=ragged_math_ops.reduce_all, rt_input=[[True, True], [True, True, False, True], [False, True]], axis=0, keepdims=True, expected=[[False, True, False, True]]), dict(ragged_reduce_op=ragged_math_ops.reduce_all, rt_input=[[True, True], [True, True, False, True], [False, True]], axis=1, keepdims=True, expected=[[True], [False], [False]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=None, keepdims=False, expected=0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9), dict(ragged_reduce_op=ragged_math_ops.reduce_prod, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=None, keepdims=False, expected=0 * 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9), dict(ragged_reduce_op=ragged_math_ops.reduce_min, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=None, keepdims=False, expected=min(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)), dict(ragged_reduce_op=ragged_math_ops.reduce_max, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=None, keepdims=False, expected=max(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)), dict(ragged_reduce_op=ragged_math_ops.reduce_mean, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=None, keepdims=False, expected=mean(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)), dict(ragged_reduce_op=ragged_math_ops.reduce_variance, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=None, keepdims=False, expected=variance(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)), dict(ragged_reduce_op=ragged_math_ops.reduce_std, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=None, keepdims=False, expected=std(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=0, keepdims=False, expected=[0 + 4 + 5 + 7 + 8, 1 + 6 + 9, 2, 3]), dict(ragged_reduce_op=ragged_math_ops.reduce_prod, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=0, keepdims=False, expected=[0 * 4 * 5 * 7 * 8, 1 * 6 * 9, 2, 3]), dict(ragged_reduce_op=ragged_math_ops.reduce_min, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=0, keepdims=False, expected=[min(0, 4, 5, 7, 8), min(1, 6, 9), 2, 3]), dict(ragged_reduce_op=ragged_math_ops.reduce_max, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=0, keepdims=False, expected=[max(0, 4, 5, 7, 8), max(1, 6, 9), 2, 3]), dict(ragged_reduce_op=ragged_math_ops.reduce_mean, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=0, keepdims=False, expected=[mean(0, 4, 5, 7, 8), mean(1, 6, 9), 2, 3]), dict(ragged_reduce_op=ragged_math_ops.reduce_variance, rt_input=[[0, 1, 2, 3], [1], [], [2, 1], [3], [4, 1]], axis=0, keepdims=False, expected=[variance(0, 1, 2, 3, 4), variance(1, 1, 1), 0, 0]), dict(ragged_reduce_op=ragged_math_ops.reduce_std, rt_input=[[1, 1, 2, 3], [1], [], [1, 1], [1], [1, 1]], axis=0, keepdims=False, expected=[std(1, 1, 1, 1, 1), std(1, 1, 1), 0, 0]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=1, keepdims=False, expected=[0 + 1 + 2 + 3, 4, 0, 5 + 6, 7, 8 + 9]), dict(ragged_reduce_op=ragged_math_ops.reduce_prod, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=1, keepdims=False, expected=[0 * 1 * 2 * 3, 4, 1, 5 * 6, 7, 8 * 9]), dict(ragged_reduce_op=ragged_math_ops.reduce_min, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=1, keepdims=False, expected=[min(0, 1, 2, 3), 4, _MAX_INT32, min(5, 6), 7, min(8, 9)]), dict(ragged_reduce_op=ragged_math_ops.reduce_max, rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]], axis=1, keepdims=False, expected=[max(0, 1, 2, 3), 4, _MIN_INT32, max(5, 6), 7, max(8, 9)]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=[], keepdims=False, expected=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=None, keepdims=False, expected=sum([1, 2, 3, 4, 5, 6, 7, 8, 9])), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=0, keepdims=False, expected=[[1 + 6 + 9, 2 + 7], [], [3 + 8, 4, 5]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=1, keepdims=False, expected=[[1 + 3, 2 + 4, 5], [6 + 8, 7], [], [9]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=2, keepdims=False, expected=[[1 + 2, 0, 3 + 4 + 5], [6 + 7, 0, 8], [], [9]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=[0, 1], keepdims=False, expected=[1 + 3 + 6 + 8 + 9, 2 + 4 + 7, 5]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=[0, 2], keepdims=False, expected=[1 + 6 + 9 + 2 + 7, 0, 3 + 8 + 4 + 5]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=[1, 2], keepdims=False, expected=[1 + 2 + 3 + 4 + 5, 6 + 7 + 8, 0, 9]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=[0, 1, 2], keepdims=False, expected=sum([1, 2, 3, 4, 5, 6, 7, 8, 9])), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=[], keepdims=True, expected=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=None, keepdims=True, expected=[[[sum([1, 2, 3, 4, 5, 6, 7, 8, 9])]]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=0, keepdims=True, expected=[[[1 + 6 + 9, 2 + 7], [], [3 + 8, 4, 5]]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=1, keepdims=True, expected=[[[1 + 3, 2 + 4, 5]], [[6 + 8, 7]], [[]], [[9]]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=2, keepdims=True, expected=[[[1 + 2], [0], [3 + 4 + 5]], [[6 + 7], [0], [8]], [], [[9]]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=[0, 1], keepdims=True, expected=[[[1 + 3 + 6 + 8 + 9, 2 + 4 + 7, 5]]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=[0, 2], keepdims=True, expected=[[[1 + 6 + 9 + 2 + 7], [0], [3 + 8 + 4 + 5]]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=[1, 2], keepdims=True, expected=[[[1 + 2 + 3 + 4 + 5]], [[6 + 7 + 8]], [[0]], [[9]]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=[0, 1, 2], keepdims=True, expected=[[[sum([1, 2, 3, 4, 5, 6, 7, 8, 9])]]]), dict(ragged_reduce_op=ragged_math_ops.reduce_mean, rt_input=[[[1, 2], [3, 4, 5]], [[6, 7], [8]], [[9]]], axis=0, keepdims=False, expected=[[mean(1, 6, 9), mean(2, 7)], [mean(3, 8), 4, 5]]), dict(ragged_reduce_op=ragged_math_ops.reduce_mean, rt_input=[[[1, 2], [3, 4, 5]], [[6, 7], [8]], [[9]]], axis=1, keepdims=False, expected=[[mean(1, 3), mean(2, 4), 5], [mean(6, 8), 7], [9]]), dict(ragged_reduce_op=ragged_math_ops.reduce_mean, rt_input=[[[1, 2], [3, 4, 5]], [[6, 7], [8]], [[9]]], axis=2, keepdims=False, expected=[[mean(1, 2), mean(3, 4, 5)], [mean(6, 7), 8], [9]]), dict(ragged_reduce_op=ragged_math_ops.reduce_variance, rt_input=[[[6, 2], [3, 4, 5]], [[6, 7], [8]], [[9]]], axis=0, keepdims=False, expected=[[variance(6, 6, 9), variance(2, 7)], [variance(3, 8), 0.0, 0.0]]), dict(ragged_reduce_op=ragged_math_ops.reduce_variance, rt_input=[[[6, 2], [3, 4, 5]], [[6, 7], [8]], [[9]]], axis=1, keepdims=False, expected=[[variance(6, 3), variance(2, 4), 0.0], [variance(6, 8), 0.0], [0.0]]), dict(ragged_reduce_op=ragged_math_ops.reduce_variance, rt_input=[[[6, 2], [6, 9, 9]], [[6, 7], [8]], [[9]]], axis=2, keepdims=False, expected=[[variance(6, 2), variance(6, 9, 9)], [variance(6, 7), 0.0], [0.0]]), dict(ragged_reduce_op=ragged_math_ops.reduce_std, rt_input=[[[6, 2], [3, 4, 5]], [[6, 7], [8]], [[9]]], axis=0, keepdims=False, expected=[[std(6, 6, 9), std(2, 7)], [std(3, 8), 0.0, 0.0]]), dict(ragged_reduce_op=ragged_math_ops.reduce_std, rt_input=[[[6, 2], [3, 4, 5]], [[6, 7], [8]], [[9]]], axis=1, keepdims=False, expected=[[std(6, 3), std(2, 4), 0.0], [std(6, 8), 0.0], [0.0]]), dict(ragged_reduce_op=ragged_math_ops.reduce_std, rt_input=[[[6, 2], [6, 9, 9]], [[6, 7], [8]], [[9]]], axis=2, keepdims=False, expected=[[std(6, 2), std(6, 9, 9)], [std(6, 7), 0.0], [0.0]]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=[-2, -1], keepdims=False, expected=[1 + 2 + 3 + 4 + 5, 6 + 7 + 8, 0, 9]), dict(ragged_reduce_op=ragged_math_ops.reduce_sum, rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]], axis=[-3, -2, -1], keepdims=False, expected=sum([1, 2, 3, 4, 5, 6, 7, 8, 9])), dict(ragged_reduce_op=ragged_math_ops.reduce_variance, rt_input=[[[0.214441], [0.214441], [0.214441], [0.214441], [0.214441], [0.214441], [0.214441]]], axis=[1], keepdims=False, expected=[[0.0]]), dict(ragged_reduce_op=ragged_math_ops.reduce_std, rt_input=[[[0.214441], [0.214441], [0.214441], [0.214441], [0.214441], [0.214441], [0.214441]]], axis=[1], keepdims=False, expected=[[0.0]]))
    def testReduce(self, ragged_reduce_op, rt_input, axis, keepdims, expected):
        if False:
            i = 10
            return i + 15
        rt_input = ragged_factory_ops.constant(rt_input)
        reduced = ragged_reduce_op(rt_input, axis, keepdims=keepdims)
        self.assertAllEqual(reduced, expected)

    def testReduceKeepsInnerDimensionShape(self):
        if False:
            while True:
                i = 10
        rt = ragged_factory_ops.constant([[[[1, 1]]]], ragged_rank=2)
        self.assertEqual(rt.shape.as_list(), [1, None, None, 2])
        reduced = ragged_math_ops.reduce_sum(rt, axis=2)
        self.assertEqual(reduced.shape.as_list(), [1, None, 2])

    def assertEqualWithNan(self, actual, expected):
        if False:
            for i in range(10):
                print('nop')
        'Like assertEqual, but NaN==NaN.'
        self.assertTrue(((actual == expected) | np.isnan(actual) & np.isnan(expected)).all())

    def testMeanNan(self):
        if False:
            return 10
        rt_as_list = [[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]]
        expected = np.array([0 + 1 + 2 + 3, 4, 0, 5 + 6, 7, 8 + 9]) / np.array([4, 1, 0, 2, 1, 2])
        rt_input = ragged_factory_ops.constant(rt_as_list)
        reduced = ragged_math_ops.reduce_mean(rt_input, axis=1)
        self.assertEqualWithNan(self.evaluate(reduced), expected)

    def testVarianceNan(self):
        if False:
            i = 10
            return i + 15
        rt_as_list = [[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]]
        expected = [variance(0, 1, 2, 3), variance(4), variance(), variance(5, 6), variance(7), variance(8, 9)]
        rt_input = ragged_factory_ops.constant(rt_as_list)
        reduced = ragged_math_ops.reduce_variance(rt_input, axis=1)
        self.assertEqualWithNan(self.evaluate(reduced), expected)

    def testStdNan(self):
        if False:
            print('Hello World!')
        rt_as_list = [[0, 1, 1, 0], [4], [], [5, 6], [7], [8, 9]]
        expected = [std(0, 1, 1, 0), std(4), std(), std(5, 6), std(7), std(8, 9)]
        rt_input = ragged_factory_ops.constant(rt_as_list)
        reduced = ragged_math_ops.reduce_std(rt_input, axis=1)
        self.assertEqualWithNan(self.evaluate(reduced), expected)

    def testMeanWithTensorInputs(self):
        if False:
            while True:
                i = 10
        tensor = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
        expected = [2.0, 20.0]
        reduced = ragged_math_ops.reduce_mean(tensor, axis=1)
        self.assertAllEqual(reduced, expected)

    def testVarianceWithTensorInputs(self):
        if False:
            return 10
        tensor = [[6.0, 9.0, 6.0], [60.0, 90.0, 60.0]]
        expected = [2.0, 200.0]
        reduced = ragged_math_ops.reduce_variance(tensor, axis=1)
        self.assertAllEqual(reduced, expected)

    def testStdWithTensorInputs(self):
        if False:
            return 10
        tensor = [[1.0, 2.0, 2.0, 1.0], [10.0, 20.0, 20.0, 10.0]]
        expected = [0.5, 5.0]
        reduced = ragged_math_ops.reduce_std(tensor, axis=1)
        self.assertAllEqual(reduced, expected)

    def testErrors(self):
        if False:
            while True:
                i = 10
        rt_input = ragged_factory_ops.constant([[1, 2, 3], [4, 5]])
        axis = array_ops.placeholder_with_default(constant_op.constant([0]), None)
        if not context.executing_eagerly():
            self.assertRaisesRegex(ValueError, 'axis must be known at graph construction time.', ragged_math_ops.reduce_sum, rt_input, axis)
        self.assertRaisesRegex(TypeError, 'axis must be an int; got str.*', ragged_math_ops.reduce_sum, rt_input, ['x'])
if __name__ == '__main__':
    googletest.main()