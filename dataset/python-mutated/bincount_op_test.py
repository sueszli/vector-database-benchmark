"""Tests for bincount using the XLA JIT."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import errors
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.platform import googletest

class BincountTest(xla_test.XLATestCase):

    def testInputRank0(self):
        if False:
            print('Hello World!')
        with self.session():
            with self.test_scope():
                bincount = gen_math_ops.bincount(arr=6, size=804, weights=[52, 351])
            with self.assertRaisesRegex(errors.InvalidArgumentError, '`weights` must be the same shape as `arr` or a length-0 `Tensor`, in which case it acts as all weights equal to 1.'):
                self.evaluate(bincount)
if __name__ == '__main__':
    googletest.main()