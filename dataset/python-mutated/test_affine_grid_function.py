import unittest
import numpy as np
import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base

class AffineGridTestCase(unittest.TestCase):

    def __init__(self, methodName='runTest', theta_shape=(20, 2, 3), output_shape=[20, 2, 5, 7], align_corners=True, dtype='float32', invalid_theta=False, variable_output_shape=False):
        if False:
            while True:
                i = 10
        super().__init__(methodName)
        self.theta_shape = theta_shape
        self.output_shape = output_shape
        self.align_corners = align_corners
        self.dtype = dtype
        self.invalid_theta = invalid_theta
        self.variable_output_shape = variable_output_shape

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.theta = np.random.randn(*self.theta_shape).astype(self.dtype)

    def base_layer(self, place):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                theta_var = paddle.static.data('input', self.theta_shape, dtype=self.dtype)
                y_var = paddle.nn.functional.affine_grid(theta_var, self.output_shape)
        feed_dict = {'input': self.theta}
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def functional(self, place):
        if False:
            print('Hello World!')
        paddle.enable_static()
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                theta_var = paddle.static.data('input', self.theta_shape, dtype=self.dtype)
                y_var = F.affine_grid(theta_var, self.output_shape, align_corners=self.align_corners)
        feed_dict = {'input': self.theta}
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def paddle_dygraph_layer(self):
        if False:
            return 10
        paddle.disable_static()
        theta_var = dg.to_variable(self.theta) if not self.invalid_theta else 'invalid'
        output_shape = dg.to_variable(self.output_shape) if self.variable_output_shape else self.output_shape
        y_var = F.affine_grid(theta_var, output_shape, align_corners=self.align_corners)
        y_np = y_var.numpy()
        return y_np

    def _test_equivalence(self, place):
        if False:
            i = 10
            return i + 15
        place = base.CPUPlace()
        result1 = self.base_layer(place)
        result2 = self.functional(place)
        result3 = self.paddle_dygraph_layer()
        if self.align_corners:
            np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def runTest(self):
        if False:
            while True:
                i = 10
        place = base.CPUPlace()
        self._test_equivalence(place)
        if base.core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            self._test_equivalence(place)

class AffineGridErrorTestCase(AffineGridTestCase):

    def runTest(self):
        if False:
            print('Hello World!')
        place = base.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(ValueError):
                self.paddle_dygraph_layer()

def add_cases(suite):
    if False:
        return 10
    suite.addTest(AffineGridTestCase(methodName='runTest'))
    suite.addTest(AffineGridTestCase(methodName='runTest', align_corners=True))
    suite.addTest(AffineGridTestCase(methodName='runTest', align_corners=False))
    suite.addTest(AffineGridTestCase(methodName='runTest', variable_output_shape=True))
    suite.addTest(AffineGridTestCase(methodName='runTest', theta_shape=(20, 2, 3), output_shape=[20, 1, 7, 7], align_corners=True))

def add_error_cases(suite):
    if False:
        return 10
    suite.addTest(AffineGridErrorTestCase(methodName='runTest', output_shape='not_valid'))
    suite.addTest(AffineGridErrorTestCase(methodName='runTest', invalid_theta=True))

def load_tests(loader, standard_tests, pattern):
    if False:
        print('Hello World!')
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite
if __name__ == '__main__':
    unittest.main()