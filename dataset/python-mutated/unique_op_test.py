"""Tests for tensorflow.kernels.unique_op."""
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test

class UniqueTest(test.TestCase):

    def testInt32(self):
        if False:
            return 10
        x = np.random.randint(2, high=10, size=7000)
        (y, idx) = array_ops.unique(x)
        (tf_y, tf_idx) = self.evaluate([y, idx])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))
        for i in range(len(x)):
            self.assertEqual(x[i], tf_y[tf_idx[i]])

    def testInt32OutIdxInt64(self):
        if False:
            print('Hello World!')
        x = np.random.randint(2, high=10, size=7000)
        (y, idx) = array_ops.unique(x, out_idx=dtypes.int64)
        (tf_y, tf_idx) = self.evaluate([y, idx])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))
        for i in range(len(x)):
            self.assertEqual(x[i], tf_y[tf_idx[i]])

    def testString(self):
        if False:
            print('Hello World!')
        indx = np.random.randint(65, high=122, size=7000)
        x = [chr(i) for i in indx]
        (y, idx) = array_ops.unique(x)
        (tf_y, tf_idx) = self.evaluate([y, idx])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))
        for i in range(len(x)):
            self.assertEqual(x[i], tf_y[tf_idx[i]].decode('ascii'))

    def testInt32Axis(self):
        if False:
            return 10
        for dtype in [np.int32, np.int64]:
            with self.subTest(dtype=dtype):
                x = np.array([[1, 0, 0], [1, 0, 0], [2, 0, 0]])
                (y0, idx0) = gen_array_ops.unique_v2(x, axis=np.array([0], dtype))
                self.assertEqual(y0.shape.rank, 2)
                (tf_y0, tf_idx0) = self.evaluate([y0, idx0])
                (y1, idx1) = gen_array_ops.unique_v2(x, axis=np.array([1], dtype))
                self.assertEqual(y1.shape.rank, 2)
                (tf_y1, tf_idx1) = self.evaluate([y1, idx1])
                self.assertAllEqual(tf_y0, np.array([[1, 0, 0], [2, 0, 0]]))
                self.assertAllEqual(tf_idx0, np.array([0, 0, 1]))
                self.assertAllEqual(tf_y1, np.array([[1, 0], [1, 0], [2, 0]]))
                self.assertAllEqual(tf_idx1, np.array([0, 1, 1]))

    def testInt32V2(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.randint(2, high=10, size=7000)
        (y, idx) = gen_array_ops.unique_v2(x, axis=np.array([], np.int32))
        (tf_y, tf_idx) = self.evaluate([y, idx])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))
        for i in range(len(x)):
            self.assertEqual(x[i], tf_y[tf_idx[i]])

    def testBool(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.choice([True, False], size=7000)
        (y, idx) = array_ops.unique(x)
        (tf_y, tf_idx) = self.evaluate([y, idx])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))
        for i in range(len(x)):
            self.assertEqual(x[i], tf_y[tf_idx[i]])

    def testBoolV2(self):
        if False:
            print('Hello World!')
        x = np.random.choice([True, False], size=7000)
        (y, idx) = gen_array_ops.unique_v2(x, axis=np.array([], np.int32))
        (tf_y, tf_idx) = self.evaluate([y, idx])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))
        for i in range(len(x)):
            self.assertEqual(x[i], tf_y[tf_idx[i]])

    @test_util.run_deprecated_v1
    def testShapeInferenceV2(self):
        if False:
            while True:
                i = 10
        'Test shape inference.'
        x = np.arange(6).reshape(3, 2, 1)
        (_, idx) = gen_array_ops.unique_v2(x, axis=[0])
        self.assertEqual(idx.shape.as_list(), [3])
        (_, idx) = gen_array_ops.unique_v2(x, axis=[1])
        self.assertEqual(idx.shape.as_list(), [2])
        (_, idx) = gen_array_ops.unique_v2(x, axis=[2])
        self.assertEqual(idx.shape.as_list(), [1])
        (_, idx) = gen_array_ops.unique_v2(x, axis=[-1])
        self.assertEqual(idx.shape.as_list(), [1])
        (_, idx) = gen_array_ops.unique_v2(x, axis=[-2])
        self.assertEqual(idx.shape.as_list(), [2])
        (_, idx) = gen_array_ops.unique_v2(x, axis=[-3])
        self.assertEqual(idx.shape.as_list(), [3])
        (_, idx) = gen_array_ops.unique_v2([0, 1, 2], axis=[])
        self.assertEqual(idx.shape.as_list(), [3])
        with self.assertRaisesRegexp(ValueError, 'axis expects a 1D vector'):
            gen_array_ops.unique_v2(x, axis=[[0]])
        with self.assertRaisesRegexp(ValueError, 'x expects a 1D vector'):
            gen_array_ops.unique_v2(x, axis=[])
        with self.assertRaisesRegexp(ValueError, 'axis does not support input tensors larger than'):
            gen_array_ops.unique_v2(x, axis=[1, 2])
        with self.assertRaisesRegexp(ValueError, 'axis expects to be in the range \\[-3, 3\\)'):
            gen_array_ops.unique_v2(x, axis=[3])
        with self.assertRaisesRegexp(ValueError, 'axis expects to be in the range \\[-3, 3\\)'):
            gen_array_ops.unique_v2(x, axis=[-4])
        x_t = array_ops.placeholder(dtypes.int32, shape=None)
        (_, idx) = gen_array_ops.unique_v2(x_t, axis=[0])
        self.assertEqual(idx.shape.as_list(), [None])
        axis_t = array_ops.placeholder(dtypes.int32, shape=None)
        (_, idx) = gen_array_ops.unique_v2(x, axis=axis_t)
        self.assertEqual(idx.shape.as_list(), [None])

    def testEmpty(self):
        if False:
            i = 10
            return i + 15
        x = np.random.randint(2, size=0)
        (y, idx) = array_ops.unique(x)
        (tf_y, tf_idx) = self.evaluate([y, idx])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))

    def testOrderedByAppearance(self):
        if False:
            print('Hello World!')
        x = np.array([3, 5, 3, 4, 1, 4, 9, 8, 6, 3, 5, 7, 8, 8, 4, 6, 4, 2, 5, 6])
        true_y = np.array([3, 5, 4, 1, 9, 8, 6, 7, 2])
        true_idx = np.array([0, 1, 0, 2, 3, 2, 4, 5, 6, 0, 1, 7, 5, 5, 2, 6, 2, 8, 1, 6])
        (y, idx) = array_ops.unique(x)
        (tf_y, tf_idx) = self.evaluate([y, idx])
        self.assertAllEqual(tf_y, true_y)
        self.assertAllEqual(tf_idx, true_idx)

class UniqueWithCountsTest(test.TestCase):

    def testInt32(self):
        if False:
            return 10
        x = np.random.randint(2, high=10, size=7000)
        (y, idx, count) = array_ops.unique_with_counts(x)
        (tf_y, tf_idx, tf_count) = self.evaluate([y, idx, count])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))
        for i in range(len(x)):
            self.assertEqual(x[i], tf_y[tf_idx[i]])
        for (value, count) in zip(tf_y, tf_count):
            self.assertEqual(count, np.sum(x == value))

    def testInt32OutIdxInt64(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.randint(2, high=10, size=7000)
        (y, idx, count) = array_ops.unique_with_counts(x, out_idx=dtypes.int64)
        (tf_y, tf_idx, tf_count) = self.evaluate([y, idx, count])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))
        for i in range(len(x)):
            self.assertEqual(x[i], tf_y[tf_idx[i]])
        for (value, count) in zip(tf_y, tf_count):
            self.assertEqual(count, np.sum(x == value))

    def testString(self):
        if False:
            print('Hello World!')
        indx = np.random.randint(65, high=122, size=7000)
        x = [chr(i) for i in indx]
        (y, idx, count) = array_ops.unique_with_counts(x)
        (tf_y, tf_idx, tf_count) = self.evaluate([y, idx, count])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))
        for i in range(len(x)):
            self.assertEqual(x[i], tf_y[tf_idx[i]].decode('ascii'))
        for (value, count) in zip(tf_y, tf_count):
            with self.subTest(value=value, count=count):
                v = [1 if x[i] == value.decode('ascii') else 0 for i in range(7000)]
                self.assertEqual(count, sum(v))

    def testInt32Axis(self):
        if False:
            while True:
                i = 10
        for dtype in [np.int32, np.int64]:
            with self.subTest(dtype=dtype):
                x = np.array([[1, 0, 0], [1, 0, 0], [2, 0, 0]])
                (y0, idx0, count0) = gen_array_ops.unique_with_counts_v2(x, axis=np.array([0], dtype))
                self.assertEqual(y0.shape.rank, 2)
                (tf_y0, tf_idx0, tf_count0) = self.evaluate([y0, idx0, count0])
                (y1, idx1, count1) = gen_array_ops.unique_with_counts_v2(x, axis=np.array([1], dtype))
                self.assertEqual(y1.shape.rank, 2)
                (tf_y1, tf_idx1, tf_count1) = self.evaluate([y1, idx1, count1])
                self.assertAllEqual(tf_y0, np.array([[1, 0, 0], [2, 0, 0]]))
                self.assertAllEqual(tf_idx0, np.array([0, 0, 1]))
                self.assertAllEqual(tf_count0, np.array([2, 1]))
                self.assertAllEqual(tf_y1, np.array([[1, 0], [1, 0], [2, 0]]))
                self.assertAllEqual(tf_idx1, np.array([0, 1, 1]))
                self.assertAllEqual(tf_count1, np.array([1, 2]))

    def testInt32V2(self):
        if False:
            print('Hello World!')
        x = np.random.randint(2, high=10, size=7000)
        (y, idx, count) = gen_array_ops.unique_with_counts_v2(x, axis=np.array([], np.int32))
        (tf_y, tf_idx, tf_count) = self.evaluate([y, idx, count])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))
        for i in range(len(x)):
            self.assertEqual(x[i], tf_y[tf_idx[i]])
        for (value, count) in zip(tf_y, tf_count):
            self.assertEqual(count, np.sum(x == value))

    def testBool(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.choice([True, False], size=7000)
        (y, idx, count) = array_ops.unique_with_counts(x)
        (tf_y, tf_idx, tf_count) = self.evaluate([y, idx, count])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))
        for i in range(len(x)):
            self.assertEqual(x[i], tf_y[tf_idx[i]])
        for (value, count) in zip(tf_y, tf_count):
            self.assertEqual(count, np.sum(x == value))

    def testBoolV2(self):
        if False:
            print('Hello World!')
        x = np.random.choice([True, False], size=7000)
        (y, idx, count) = gen_array_ops.unique_with_counts_v2(x, axis=np.array([], np.int32))
        (tf_y, tf_idx, tf_count) = self.evaluate([y, idx, count])
        self.assertEqual(len(x), len(tf_idx))
        self.assertEqual(len(tf_y), len(np.unique(x)))
        for i in range(len(x)):
            self.assertEqual(x[i], tf_y[tf_idx[i]])
        for (value, count) in zip(tf_y, tf_count):
            self.assertEqual(count, np.sum(x == value))

    def testFloat(self):
        if False:
            return 10
        x = [0.0, 1.0, np.nan, np.nan]
        (y, idx, count) = gen_array_ops.unique_with_counts_v2(x, axis=np.array([], np.int32))
        (tf_y, tf_idx, tf_count) = self.evaluate([y, idx, count])
        self.assertEqual(len(x), len(tf_idx))
        for i in range(len(x)):
            if np.isnan(x[i]):
                self.assertTrue(np.isnan(tf_y[tf_idx[i]]))
            else:
                self.assertEqual(x[i], tf_y[tf_idx[i]])
        for (value, count) in zip(tf_y, tf_count):
            if np.isnan(value):
                self.assertEqual(count, 1)
            else:
                self.assertEqual(count, np.sum(x == value))

    def testEmpty(self):
        if False:
            i = 10
            return i + 15
        x = np.random.randint(2, size=0)
        (y, idx, count) = array_ops.unique_with_counts(x)
        (tf_y, tf_idx, tf_count) = self.evaluate([y, idx, count])
        self.assertEqual(tf_idx.shape, (0,))
        self.assertEqual(tf_y.shape, (0,))
        self.assertEqual(tf_count.shape, (0,))

    def testOrderedByAppearance(self):
        if False:
            return 10
        x = np.array([3, 5, 3, 4, 1, 4, 9, 8, 6, 3, 5, 7, 8, 8, 4, 6, 4, 2, 5, 6])
        true_y = np.array([3, 5, 4, 1, 9, 8, 6, 7, 2])
        true_idx = np.array([0, 1, 0, 2, 3, 2, 4, 5, 6, 0, 1, 7, 5, 5, 2, 6, 2, 8, 1, 6])
        true_count = np.array([3, 3, 4, 1, 1, 3, 3, 1, 1])
        (y, idx, count) = array_ops.unique_with_counts(x)
        (tf_y, tf_idx, tf_count) = self.evaluate([y, idx, count])
        self.assertAllEqual(tf_y, true_y)
        self.assertAllEqual(tf_idx, true_idx)
        self.assertAllEqual(tf_count, true_count)
if __name__ == '__main__':
    test.main()