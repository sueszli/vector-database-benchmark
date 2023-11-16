import unittest
import numpy as np
import paddle.base.dygraph as dg
from paddle import base

class TestComplexGetitemLayer(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            self._places.append(base.CUDAPlace(0))

    def test_case1(self):
        if False:
            for i in range(10):
                print('nop')
        x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
        x_np_slice = x_np[0]
        for place in self._places:
            with dg.guard(place):
                x_var = dg.to_variable(x_np)
                x_var_slice = x_var[0]
            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)

    def test_case2(self):
        if False:
            print('Hello World!')
        x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
        x_np_slice = x_np[0][1]
        for place in self._places:
            with dg.guard(place):
                x_var = dg.to_variable(x_np)
                x_var_slice = x_var[0][1]
            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)

    def test_case3(self):
        if False:
            while True:
                i = 10
        x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
        x_np_slice = x_np[0][1][2]
        for place in self._places:
            with dg.guard(place):
                x_var = dg.to_variable(x_np)
                x_var_slice = x_var[0][1][2]
            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)

    def test_case4(self):
        if False:
            return 10
        x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
        x_np_slice = x_np[0][1][0:3]
        for place in self._places:
            with dg.guard(place):
                x_var = dg.to_variable(x_np)
                x_var_slice = x_var[0][1][0:3]
            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)

    def test_case5(self):
        if False:
            return 10
        x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
        x_np_slice = x_np[0][1][0:4:2]
        for place in self._places:
            with dg.guard(place):
                x_var = dg.to_variable(x_np)
                x_var_slice = x_var[0][1][0:4:2]
            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)
            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)

    def test_case6(self):
        if False:
            i = 10
            return i + 15
        x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
        x_np_slice = x_np[0][1:3][0:4:2]
        for place in self._places:
            with dg.guard(place):
                x_var = dg.to_variable(x_np)
                x_var_slice = x_var[0][1:3][0:4:2]
            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)
if __name__ == '__main__':
    unittest.main()