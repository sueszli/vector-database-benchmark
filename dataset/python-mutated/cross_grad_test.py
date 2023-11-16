"""Tests for tensorflow.ops.nn_ops.Cross."""
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class CrossOpTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testGradientRandomValues(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            us = [2, 3]
            u = array_ops.reshape([0.854, -0.616, 0.767, 0.725, -0.927, 0.159], shape=us)
            v = array_ops.reshape([-0.522, 0.755, 0.407, -0.652, 0.241, 0.247], shape=us)
            s = math_ops.cross(u, v)
            (jacob_u, jacob_v) = gradient_checker.compute_gradient([u, v], [us, us], s, us)
        self.assertAllClose(jacob_u[0], jacob_u[1], rtol=0.001, atol=0.001)
        self.assertAllClose(jacob_v[0], jacob_v[1], rtol=0.001, atol=0.001)
if __name__ == '__main__':
    test.main()