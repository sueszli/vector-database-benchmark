import unittest
import numpy as np
import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base

class GridSampleTestCase(unittest.TestCase):

    def __init__(self, methodName='runTest', x_shape=[2, 2, 3, 3], grid_shape=[2, 3, 3, 2], mode='bilinear', padding_mode='zeros', align_corners=False):
        if False:
            print('Hello World!')
        super().__init__(methodName)
        self.padding_mode = padding_mode
        self.x_shape = x_shape
        self.grid_shape = grid_shape
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = 'float64'

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = np.random.randn(*self.x_shape).astype(self.dtype)
        self.grid = np.random.uniform(-1, 1, self.grid_shape).astype(self.dtype)

    def static_functional(self, place):
        if False:
            while True:
                i = 10
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                x = paddle.static.data('x', self.x_shape, dtype=self.dtype)
                grid = paddle.static.data('grid', self.grid_shape, dtype=self.dtype)
                y_var = F.grid_sample(x, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)
        feed_dict = {'x': self.x, 'grid': self.grid}
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def dynamic_functional(self):
        if False:
            i = 10
            return i + 15
        x_t = paddle.to_tensor(self.x)
        grid_t = paddle.to_tensor(self.grid)
        y_t = F.grid_sample(x_t, grid_t, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)
        y_np = y_t.numpy()
        return y_np

    def _test_equivalence(self, place):
        if False:
            while True:
                i = 10
        result1 = self.static_functional(place)
        with dg.guard(place):
            result2 = self.dynamic_functional()
        np.testing.assert_array_almost_equal(result1, result2)

    def runTest(self):
        if False:
            print('Hello World!')
        place = base.CPUPlace()
        self._test_equivalence(place)
        if base.core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            self._test_equivalence(place)

class GridSampleErrorTestCase(GridSampleTestCase):

    def runTest(self):
        if False:
            i = 10
            return i + 15
        place = base.CPUPlace()
        with self.assertRaises(ValueError):
            self.static_functional(place)

def add_cases(suite):
    if False:
        print('Hello World!')
    suite.addTest(GridSampleTestCase(methodName='runTest'))
    suite.addTest(GridSampleTestCase(methodName='runTest', mode='bilinear', padding_mode='reflection', align_corners=True))
    suite.addTest(GridSampleTestCase(methodName='runTest', mode='bilinear', padding_mode='zeros', align_corners=True))

def add_error_cases(suite):
    if False:
        while True:
            i = 10
    suite.addTest(GridSampleErrorTestCase(methodName='runTest', padding_mode='VALID'))
    suite.addTest(GridSampleErrorTestCase(methodName='runTest', align_corners='VALID'))
    suite.addTest(GridSampleErrorTestCase(methodName='runTest', mode='VALID'))

def load_tests(loader, standard_tests, pattern):
    if False:
        for i in range(10):
            print('nop')
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite

class TestGridSampleAPI(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            x = paddle.randn([1, 1, 3, 3])
            F.grid_sample(x, 1.0)
        with self.assertRaises(ValueError):
            x = paddle.randn([1, 1, 3, 3])
            F.grid_sample(1.0, x)
if __name__ == '__main__':
    unittest.main()