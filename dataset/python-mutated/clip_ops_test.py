"""Tests for tensorflow.ops.clip_ops."""
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class ClipTest(test.TestCase):

    def testClipByValue(self):
        if False:
            print('Hello World!')
        with self.session():
            x = constant_op.constant([-5.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
            np_ans = [[-4.4, 2.0, 3.0], [4.0, 4.4, 4.4]]
            clip_value = 4.4
            ans = clip_ops.clip_by_value(x, -clip_value, clip_value)
            tf_ans = self.evaluate(ans)
        self.assertAllClose(np_ans, tf_ans)

    def testClipByValue0Type(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8]:
            with self.cached_session():
                x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
                np_ans = [[2, 2, 3], [4, 4, 4]]
                clip_value_min = 2
                clip_value_max = 4
                ans = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
                tf_ans = self.evaluate(ans)
            self.assertAllClose(np_ans, tf_ans)

    def testClipByValue1Type(self):
        if False:
            i = 10
            return i + 15
        for dtype in [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8]:
            with self.cached_session():
                x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
                np_ans = [[2, 2, 3], [4, 4, 4]]
                clip_value_min = constant_op.constant([2, 2, 2, 3, 3, 3], shape=[2, 3], dtype=dtype)
                clip_value_max = 4
                ans = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
                tf_ans = self.evaluate(ans)
            self.assertAllClose(np_ans, tf_ans)

    def testClipByValue2Type(self):
        if False:
            while True:
                i = 10
        for dtype in [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8]:
            with self.cached_session():
                x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
                np_ans = [[4, 4, 4], [4, 5, 6]]
                clip_value_min = 4
                clip_value_max = constant_op.constant([6, 6, 6, 6, 6, 6], shape=[2, 3], dtype=dtype)
                ans = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
                tf_ans = self.evaluate(ans)
            self.assertAllClose(np_ans, tf_ans)

    def testClipByValue3Type(self):
        if False:
            return 10
        for dtype in [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8]:
            with self.cached_session():
                x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
                np_ans = [[2, 2, 3], [5, 5, 6]]
                clip_value_min = constant_op.constant([2, 2, 2, 5, 5, 5], shape=[2, 3], dtype=dtype)
                clip_value_max = constant_op.constant([5, 5, 5, 7, 7, 7], shape=[2, 3], dtype=dtype)
                ans = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
                tf_ans = self.evaluate(ans)
            self.assertAllClose(np_ans, tf_ans)

    def testClipByValueGradient(self):
        if False:
            for i in range(10):
                print('nop')

        def grad(x, y, z, clip_fn):
            if False:
                i = 10
                return i + 15
            x = constant_op.constant(x, dtype=dtypes.float32)
            y = constant_op.constant(y, dtype=dtypes.float32)
            z = constant_op.constant(z, dtype=dtypes.float32)
            with backprop.GradientTape() as tape:
                tape.watch(x)
                tape.watch(y)
                tape.watch(z)
                output = clip_fn(x, y, z)
            return tape.gradient(output, [x, y, z])
        for f in (clip_ops.clip_by_value, gen_math_ops._clip_by_value):
            with self.subTest(f=f):
                (xg, yg, zg) = grad(0, -1, 1, clip_fn=f)
                self.assertEqual(self.evaluate(xg), 1)
                self.assertEqual(self.evaluate(yg), 0)
                self.assertEqual(self.evaluate(zg), 0)
                (xg, yg, zg) = grad(2, -1, 1, clip_fn=f)
                self.assertEqual(self.evaluate(xg), 0)
                self.assertEqual(self.evaluate(yg), 0)
                self.assertEqual(self.evaluate(zg), 1)
                (xg, yg, zg) = grad([0, -2, 2, -2], -1, 1, clip_fn=f)
                self.assertAllEqual(self.evaluate(xg), [1, 0, 0, 0])
                self.assertEqual(self.evaluate(yg), 2)
                self.assertEqual(self.evaluate(zg), 1)
                (xg, yg, zg) = grad([-1, -2, 0, 2], [-2, -1, -3, 0], 1, clip_fn=f)
                self.assertAllEqual(self.evaluate(xg), [1, 0, 1, 0])
                self.assertAllEqual(self.evaluate(yg), [0, 1, 0, 0])
                self.assertEqual(self.evaluate(zg), 1)
                (xg, yg, zg) = grad([-1, -2, 0, 2], [-2, -1, -3, 0], [1, 2, -1, 1], clip_fn=f)
                self.assertAllEqual(self.evaluate(xg), [1, 0, 0, 0])
                self.assertAllEqual(self.evaluate(yg), [0, 1, 0, 0])
                self.assertAllEqual(self.evaluate(zg), [0, 0, 1, 1])
        (xg, yg, zg) = grad([[-2, 3], [2, -1]], [-1, -2], [[1, 2], [3, 4]], clip_fn=clip_ops.clip_by_value)
        self.assertAllEqual(self.evaluate(xg), [[0, 0], [1, 1]])
        self.assertAllEqual(self.evaluate(yg), [1, 0])
        self.assertAllEqual(self.evaluate(zg), [[0, 1], [0, 0]])

    def testClipByValueBadShape(self):
        if False:
            i = 10
            return i + 15
        with self.session():
            x = constant_op.constant([-5.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3, 1])
            clip = constant_op.constant([1.0, 2.0])
            with self.assertRaises(ValueError):
                _ = clip_ops.clip_by_value(x, -clip, clip)
            with self.assertRaises(ValueError):
                _ = clip_ops.clip_by_value(x, 1.0, clip)

    def testClipByValueNonFinite(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            x = constant_op.constant([float('NaN'), float('Inf'), -float('Inf')])
            np_ans = [float('NaN'), 4.0, -4.0]
            clip_value = 4.0
            ans = clip_ops.clip_by_value(x, -clip_value, clip_value)
            tf_ans = self.evaluate(ans)
        self.assertAllClose(np_ans, tf_ans)

    def _testClipIndexedSlicesByValue(self, values, indices, shape, clip_value_min, clip_value_max, expected):
        if False:
            print('Hello World!')
        with self.session():
            values = constant_op.constant(values)
            indices = constant_op.constant(indices)
            shape = constant_op.constant(shape)
            indexed_slices = indexed_slices_lib.IndexedSlices(values, indices, shape)
            clipped = clip_ops.clip_by_value(indexed_slices, clip_value_min, clip_value_max)
            self.assertIsInstance(clipped, indexed_slices_lib.IndexedSlices)
        self.assertAllClose(clipped.values, expected)

    def testClipByValueWithIndexedSlicesClipped(self):
        if False:
            while True:
                i = 10
        values = [[[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], [[0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]]
        indices = [2, 6]
        shape = [10, 2, 3]
        self._testClipIndexedSlicesByValue(values, indices, shape, -2.0, 2.0, [[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]])
        self._testClipIndexedSlicesByValue(values, indices, shape, 1.0, 2.0, [[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], [[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]])
        self._testClipIndexedSlicesByValue(values, indices, shape, -2.0, -1.0, [[[-2.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]])

    def testClipByNormClipped(self):
        if False:
            i = 10
            return i + 15
        with self.session():
            x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
            np_ans = [[-2.4, 0.0, 0.0], [3.2, 0.0, 0.0]]
            clip_norm = 4.0
            ans = clip_ops.clip_by_norm(x, clip_norm)
            tf_ans = self.evaluate(ans)
            ans = clip_ops.clip_by_norm(x, clip_norm)
            tf_ans_tensor = self.evaluate(ans)
        self.assertAllClose(np_ans, tf_ans)
        self.assertAllClose(np_ans, tf_ans_tensor)

    @test_util.run_deprecated_v1
    def testClipByNormGradientZeros(self):
        if False:
            while True:
                i = 10
        with self.session():
            x = array_ops.zeros([3])
            b = clip_ops.clip_by_norm(x, 1.0)
            (grad,) = gradients_impl.gradients(b, x)
            self.assertAllEqual(grad, [1.0, 1.0, 1.0])

    def testClipByNormBadShape(self):
        if False:
            i = 10
            return i + 15
        with self.session():
            x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3, 1])
            clip = constant_op.constant([1.0, 2.0])
            with self.assertRaises(ValueError):
                _ = clip_ops.clip_by_norm(x, clip)

    def testClipByNormNotClipped(self):
        if False:
            print('Hello World!')
        with self.session():
            x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
            np_ans = [[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
            clip_norm = 6.0
            ans = clip_ops.clip_by_norm(x, clip_norm)
            tf_ans = self.evaluate(ans)
        self.assertAllClose(np_ans, tf_ans)

    def testClipByNormZero(self):
        if False:
            return 10
        with self.session():
            x = constant_op.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=[2, 3])
            np_ans = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            clip_norm = 6.0
            ans = clip_ops.clip_by_norm(x, clip_norm)
            tf_ans = self.evaluate(ans)
        self.assertAllClose(np_ans, tf_ans)

    def testClipByNormClippedWithDim0(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 3.0], shape=[2, 3])
            np_ans = [[-2.4, 0.0, 0.0], [3.2, 0.0, 3.0]]
            clip_norm = 4.0
            ans = clip_ops.clip_by_norm(x, clip_norm, [0])
            tf_ans = self.evaluate(ans)
        self.assertAllClose(np_ans, tf_ans)

    def testClipByNormClippedWithDim1(self):
        if False:
            i = 10
            return i + 15
        with self.session():
            x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 3.0], shape=[2, 3])
            np_ans = [[-3.0, 0.0, 0.0], [3.2, 0.0, 2.4]]
            clip_norm = 4.0
            ans = clip_ops.clip_by_norm(x, clip_norm, [1])
            tf_ans = self.evaluate(ans)
        self.assertAllClose(np_ans, tf_ans)

    def testClipByNormNotClippedWithAxes(self):
        if False:
            while True:
                i = 10
        with self.session():
            x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 3.0], shape=[2, 3])
            np_ans = [[-3.0, 0.0, 0.0], [4.0, 0.0, 3.0]]
            clip_norm = 6.0
            ans = clip_ops.clip_by_norm(x, clip_norm, [1])
            tf_ans = self.evaluate(ans)
        self.assertAllClose(np_ans, tf_ans)

    def testClipByGlobalNormClipped(self):
        if False:
            return 10
        with self.session():
            x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
            x1 = constant_op.constant([1.0, -2.0])
            clip_norm = 4.0
            np_ans_0 = [[-1.6, 0.0, 0.0], [3.2, 0.0, 0.0]]
            np_ans_1 = [0.8, -1.6]
            (ans, norm) = clip_ops.clip_by_global_norm((x0, x1), clip_norm)
            tf_ans_1 = self.evaluate(ans[0])
            tf_ans_2 = self.evaluate(ans[1])
            tf_norm = self.evaluate(norm)
        self.assertAllClose(tf_norm, 5.0)
        self.assertAllClose(np_ans_0, tf_ans_1)
        self.assertAllClose(np_ans_1, tf_ans_2)

    def testClipByGlobalNormClippedTensor(self):
        if False:
            while True:
                i = 10
        with self.session():
            x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
            x1 = constant_op.constant([1.0, -2.0])
            clip_norm = constant_op.constant(4.0)
            np_ans_0 = [[-1.6, 0.0, 0.0], [3.2, 0.0, 0.0]]
            np_ans_1 = [0.8, -1.6]
            (ans, norm) = clip_ops.clip_by_global_norm((x0, x1), clip_norm)
            tf_ans_1 = self.evaluate(ans[0])
            tf_ans_2 = self.evaluate(ans[1])
            tf_norm = self.evaluate(norm)
        self.assertAllClose(tf_norm, 5.0)
        self.assertAllClose(np_ans_0, tf_ans_1)
        self.assertAllClose(np_ans_1, tf_ans_2)

    def testClipByGlobalNormSupportsNone(self):
        if False:
            i = 10
            return i + 15
        with self.session():
            x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
            x1 = constant_op.constant([1.0, -2.0])
            clip_norm = 4.0
            np_ans_0 = [[-1.6, 0.0, 0.0], [3.2, 0.0, 0.0]]
            np_ans_1 = [0.8, -1.6]
            (ans, norm) = clip_ops.clip_by_global_norm((x0, None, x1, None), clip_norm)
            self.assertTrue(ans[1] is None)
            self.assertTrue(ans[3] is None)
            tf_ans_1 = self.evaluate(ans[0])
            tf_ans_2 = self.evaluate(ans[2])
            tf_norm = self.evaluate(norm)
        self.assertAllClose(tf_norm, 5.0)
        self.assertAllClose(np_ans_0, tf_ans_1)
        self.assertAllClose(np_ans_1, tf_ans_2)

    @test_util.run_deprecated_v1
    def testClipByGlobalNormWithIndexedSlicesClipped(self):
        if False:
            while True:
                i = 10
        with self.session():
            x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
            x1 = indexed_slices_lib.IndexedSlices(constant_op.constant([1.0, -2.0]), constant_op.constant([3, 4]))
            clip_norm = 4.0
            np_ans_0 = [[-1.6, 0.0, 0.0], [3.2, 0.0, 0.0]]
            np_ans_1 = [0.8, -1.6]
            (ans, norm) = clip_ops.clip_by_global_norm([x0, x1], clip_norm)
            tf_ans_1 = self.evaluate(ans[0])
            tf_ans_2 = self.evaluate(ans[1].values)
            tf_norm = self.evaluate(norm)
        self.assertAllClose(tf_norm, 5.0)
        self.assertAllClose(np_ans_0, tf_ans_1)
        self.assertAllClose(np_ans_1, tf_ans_2)

    def testClipByGlobalNormPreservesDenseShape(self):
        if False:
            while True:
                i = 10
        dense_shape = (1,)
        slices = indexed_slices_lib.IndexedSlices(constant_op.constant([1.0]), constant_op.constant([0]), dense_shape=dense_shape)
        (ans, _) = clip_ops.clip_by_global_norm([slices], 1.0)
        modified_slices = ans[0]
        self.assertEqual(dense_shape, slices.dense_shape)
        self.assertEqual(dense_shape, modified_slices.dense_shape)

    def testClipByGlobalNormNotClipped(self):
        if False:
            print('Hello World!')
        with self.session():
            x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
            x1 = constant_op.constant([1.0, -2.0])
            np_ans_0 = [[-2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
            np_ans_1 = [1.0, -2.0]
            clip_norm = 6.0
            (ans, norm) = clip_ops.clip_by_global_norm([x0, x1], clip_norm)
            tf_ans_1 = self.evaluate(ans[0])
            tf_ans_2 = self.evaluate(ans[1])
            tf_norm = self.evaluate(norm)
        self.assertAllClose(tf_norm, 5.0)
        self.assertAllClose(np_ans_0, tf_ans_1)
        self.assertAllClose(np_ans_1, tf_ans_2)

    def testClipByGlobalNormZero(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            x0 = constant_op.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=[2, 3])
            x1 = constant_op.constant([0.0, 0.0])
            np_ans_0 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            np_ans_1 = [0.0, 0.0]
            clip_norm = 6.0
            (ans, norm) = clip_ops.clip_by_global_norm([x0, x1], clip_norm)
            tf_ans_1 = self.evaluate(ans[0])
            tf_ans_2 = self.evaluate(ans[1])
            tf_norm = self.evaluate(norm)
        self.assertAllClose(tf_norm, 0.0)
        self.assertAllClose(np_ans_0, tf_ans_1)
        self.assertAllClose(np_ans_1, tf_ans_2)

    def testClipByGlobalNormInf(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            x0 = constant_op.constant([-2.0, 0.0, np.inf, 4.0, 0.0, 0.0], shape=[2, 3])
            x1 = constant_op.constant([1.0, -2.0])
            clip_norm = 6.0
            (ans, norm) = clip_ops.clip_by_global_norm([x0, x1], clip_norm)
            tf_ans_1 = self.evaluate(ans[0])
            tf_ans_2 = self.evaluate(ans[1])
            tf_norm = self.evaluate(norm)
            self.assertAllEqual(tf_norm, float('inf'))
            self.assertAllEqual(tf_ans_1, np.full([2, 3], float('nan')))
            self.assertAllEqual(tf_ans_2, np.full([2], float('nan')))

    def testClipByAverageNormClipped(self):
        if False:
            print('Hello World!')
        with self.session():
            x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
            np_ans = [[-2.88, 0.0, 0.0], [3.84, 0.0, 0.0]]
            clip_norm = 0.8
            ans = clip_ops.clip_by_average_norm(x, clip_norm)
            tf_ans = self.evaluate(ans)
        self.assertAllClose(np_ans, tf_ans)

    def testClipByAverageNormClippedTensor(self):
        if False:
            return 10
        with self.session():
            x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
            np_ans = [[-2.88, 0.0, 0.0], [3.84, 0.0, 0.0]]
            clip_norm = constant_op.constant(0.8)
            ans = clip_ops.clip_by_average_norm(x, clip_norm)
            tf_ans = self.evaluate(ans)
        self.assertAllClose(np_ans, tf_ans)

    def testClipByAverageNormNotClipped(self):
        if False:
            print('Hello World!')
        with self.session():
            x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
            np_ans = [[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
            clip_norm = 0.9
            ans = clip_ops.clip_by_average_norm(x, clip_norm)
            tf_ans = self.evaluate(ans)
        self.assertAllClose(np_ans, tf_ans)

    def testClipByAverageNormZero(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            x = constant_op.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=[2, 3])
            np_ans = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            clip_norm = 0.9
            ans = clip_ops.clip_by_average_norm(x, clip_norm)
            tf_ans = self.evaluate(ans)
        self.assertAllClose(np_ans, tf_ans)

    def testClipByAverageNormReplacedWithClipByNorm(self):
        if False:
            i = 10
            return i + 15
        with self.session():
            x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
            clip_norm = constant_op.constant(0.8)
            with_norm = clip_ops.clip_by_average_norm(x, clip_norm)
            without_norm = clip_ops.clip_by_norm(x, clip_norm * math_ops.cast(array_ops.size(x), dtypes.float32))
            clip_by_average_norm_ans = self.evaluate(with_norm)
            clip_by_norm_ans = self.evaluate(without_norm)
            self.assertAllClose(clip_by_average_norm_ans, clip_by_norm_ans)

    @test_util.run_deprecated_v1
    def testClipByValueEmptyTensor(self):
        if False:
            for i in range(10):
                print('nop')
        zero = array_ops.placeholder(dtype=dtypes.float32, shape=None)
        x = clip_ops.clip_by_value(zero, zero, zero)
        y = clip_ops.clip_by_value(zero, 1.0, 1.0)
        z = clip_ops.clip_by_value(zero, zero, 1.0)
        w = clip_ops.clip_by_value(zero, 1.0, zero)
        with self.session() as sess:
            sess.run([x, y, z, w], feed_dict={zero: np.zeros((7, 0))})
if __name__ == '__main__':
    test.main()