"""Functional tests for XLA Reverse Ops."""
import itertools
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest

class ReverseOpsTest(xla_test.XLATestCase):

    def testReverseOneDim(self):
        if False:
            print('Hello World!')
        shape = (7, 5, 9, 11)
        for revdim in range(-len(shape), len(shape)):
            self._AssertReverseEqual([revdim], shape)

    def testReverseMoreThanOneDim(self):
        if False:
            i = 10
            return i + 15
        shape = (7, 5, 9, 11)
        for revdims in itertools.chain.from_iterable((itertools.combinations(range(-offset, len(shape) - offset), k) for k in range(2, len(shape) + 1) for offset in range(0, len(shape)))):
            self._AssertReverseEqual(revdims, shape)

    def _AssertReverseEqual(self, revdims, shape):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(120)
        pval = np.random.randint(0, 100, size=shape).astype(float)
        with self.session():
            with self.test_scope():
                p = array_ops.placeholder(dtypes.int32, shape=shape)
                axis = constant_op.constant(np.array(revdims, dtype=np.int32), shape=(len(revdims),), dtype=dtypes.int32)
                rval = array_ops.reverse(p, axis).eval({p: pval})
                slices = tuple((slice(-1, None, -1) if d in revdims or d - len(shape) in revdims else slice(None) for d in range(len(shape))))
            self.assertEqual(pval[slices].flatten().tolist(), rval.flatten().tolist())
if __name__ == '__main__':
    googletest.main()