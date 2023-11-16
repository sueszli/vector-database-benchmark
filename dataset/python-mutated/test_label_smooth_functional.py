import unittest
import numpy as np
import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base
from paddle.pir_utils import test_with_pir_api

class LabelSmoothTestCase(unittest.TestCase):

    def __init__(self, methodName='runTest', label_shape=(20, 1), prior_dist=None, epsilon=0.1, dtype='float32'):
        if False:
            print('Hello World!')
        super().__init__(methodName)
        self.label_shape = label_shape
        self.prior_dist = prior_dist
        self.dtype = dtype
        self.epsilon = epsilon

    def setUp(self):
        if False:
            print('Hello World!')
        self.label = np.random.randn(*self.label_shape).astype(self.dtype)

    def base_layer(self, place):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                label_var = paddle.static.data('input', self.label_shape, dtype=self.dtype)
                y_var = F.label_smooth(label_var, prior_dist=self.prior_dist, epsilon=self.epsilon)
        feed_dict = {'input': self.label}
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def functional(self, place):
        if False:
            return 10
        paddle.enable_static()
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                label_var = paddle.static.data('input', self.label_shape, dtype=self.dtype)
                y_var = F.label_smooth(label_var, prior_dist=self.prior_dist, epsilon=self.epsilon)
        feed_dict = {'input': self.label}
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def paddle_dygraph_layer(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        label_var = dg.to_variable(self.label)
        y_var = F.label_smooth(label_var, prior_dist=self.prior_dist, epsilon=self.epsilon)
        y_np = y_var.numpy()
        return y_np

    @test_with_pir_api
    def _test_equivalence(self, place):
        if False:
            while True:
                i = 10
        place = base.CPUPlace()
        result1 = self.base_layer(place)
        result2 = self.functional(place)
        result3 = self.paddle_dygraph_layer()
        np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def runTest(self):
        if False:
            print('Hello World!')
        place = base.CPUPlace()
        self._test_equivalence(place)
        if base.core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            self._test_equivalence(place)

class LabelSmoothErrorTestCase(LabelSmoothTestCase):

    def runTest(self):
        if False:
            for i in range(10):
                print('nop')
        place = base.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(ValueError):
                self.paddle_dygraph_layer()

def add_cases(suite):
    if False:
        print('Hello World!')
    suite.addTest(LabelSmoothTestCase(methodName='runTest'))
    suite.addTest(LabelSmoothTestCase(methodName='runTest', label_shape=[2, 3, 1]))

def add_error_cases(suite):
    if False:
        i = 10
        return i + 15
    suite.addTest(LabelSmoothErrorTestCase(methodName='runTest', epsilon=2))

def load_tests(loader, standard_tests, pattern):
    if False:
        while True:
            i = 10
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite
if __name__ == '__main__':
    unittest.main()