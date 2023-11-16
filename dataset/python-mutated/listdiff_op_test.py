"""Tests for XLA listdiff operator."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class ListDiffTest(xla_test.XLATestCase):

    def _testListDiff(self, x, y, out, idx):
        if False:
            while True:
                i = 10
        for dtype in [dtypes.int32, dtypes.int64]:
            for index_dtype in [dtypes.int32, dtypes.int64]:
                with self.session():
                    x_tensor = ops.convert_to_tensor(x, dtype=dtype)
                    y_tensor = ops.convert_to_tensor(y, dtype=dtype)
                    with self.test_scope():
                        (out_tensor, idx_tensor) = array_ops.listdiff(x_tensor, y_tensor, out_idx=index_dtype)
                        (tf_out, tf_idx) = self.evaluate([out_tensor, idx_tensor])
                self.assertAllEqual(out, tf_out)
                self.assertAllEqual(idx, tf_idx)
                self.assertEqual(1, out_tensor.get_shape().ndims)
                self.assertEqual(1, idx_tensor.get_shape().ndims)

    def testBasic1(self):
        if False:
            return 10
        self._testListDiff(x=[1, 2, 3, 4], y=[1, 2], out=[3, 4], idx=[2, 3])

    def testBasic2(self):
        if False:
            return 10
        self._testListDiff(x=[1, 2, 3, 4], y=[2], out=[1, 3, 4], idx=[0, 2, 3])

    def testBasic3(self):
        if False:
            while True:
                i = 10
        self._testListDiff(x=[1, 4, 3, 2], y=[4, 2], out=[1, 3], idx=[0, 2])

    def testDuplicates(self):
        if False:
            print('Hello World!')
        self._testListDiff(x=[1, 2, 4, 3, 2, 3, 3, 1], y=[4, 2], out=[1, 3, 3, 3, 1], idx=[0, 3, 5, 6, 7])

    def testRandom(self):
        if False:
            while True:
                i = 10
        num_random_tests = 10
        int_low = -7
        int_high = 8
        max_size = 50
        for _ in range(num_random_tests):
            x_size = np.random.randint(max_size + 1)
            x = np.random.randint(int_low, int_high, size=x_size)
            y_size = np.random.randint(max_size + 1)
            y = np.random.randint(int_low, int_high, size=y_size)
            out_idx = [(entry, pos) for (pos, entry) in enumerate(x) if entry not in y]
            if out_idx:
                (out, idx) = map(list, zip(*out_idx))
            else:
                out = []
                idx = []
            self._testListDiff(list(x), list(y), out, idx)

    def testFullyOverlapping(self):
        if False:
            for i in range(10):
                print('nop')
        self._testListDiff(x=[1, 2, 3, 4], y=[1, 2, 3, 4], out=[], idx=[])

    def testNonOverlapping(self):
        if False:
            for i in range(10):
                print('nop')
        self._testListDiff(x=[1, 2, 3, 4], y=[5, 6], out=[1, 2, 3, 4], idx=[0, 1, 2, 3])

    def testEmptyX(self):
        if False:
            for i in range(10):
                print('nop')
        self._testListDiff(x=[], y=[1, 2], out=[], idx=[])

    def testEmptyY(self):
        if False:
            print('Hello World!')
        self._testListDiff(x=[1, 2, 3, 4], y=[], out=[1, 2, 3, 4], idx=[0, 1, 2, 3])

    def testEmptyXY(self):
        if False:
            for i in range(10):
                print('nop')
        self._testListDiff(x=[], y=[], out=[], idx=[])
if __name__ == '__main__':
    test.main()