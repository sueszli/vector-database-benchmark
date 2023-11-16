import unittest
import numpy as np
import paddle
import paddle.base.dygraph as dg
from paddle import base

class ComplexKronTestCase(unittest.TestCase):

    def __init__(self, methodName='runTest', x=None, y=None):
        if False:
            while True:
                i = 10
        super().__init__(methodName)
        self.x = x
        self.y = y

    def setUp(self):
        if False:
            print('Hello World!')
        self.ref_result = np.kron(self.x, self.y)
        self._places = [paddle.CPUPlace()]
        if base.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def runTest(self):
        if False:
            print('Hello World!')
        for place in self._places:
            self.test_kron_api(place)

    def test_kron_api(self, place):
        if False:
            while True:
                i = 10
        with dg.guard(place):
            x_var = dg.to_variable(self.x)
            y_var = dg.to_variable(self.y)
            out_var = paddle.kron(x_var, y_var)
            np.testing.assert_allclose(out_var.numpy(), self.ref_result, rtol=1e-05)

def load_tests(loader, standard_tests, pattern):
    if False:
        return 10
    suite = unittest.TestSuite()
    for dtype in ['float32', 'float64']:
        suite.addTest(ComplexKronTestCase(x=np.random.randn(2, 2).astype(dtype) + 1j * np.random.randn(2, 2).astype(dtype), y=np.random.randn(3, 3).astype(dtype) + 1j * np.random.randn(3, 3).astype(dtype)))
        suite.addTest(ComplexKronTestCase(x=np.random.randn(2, 2).astype(dtype), y=np.random.randn(3, 3).astype(dtype) + 1j * np.random.randn(3, 3).astype(dtype)))
        suite.addTest(ComplexKronTestCase(x=np.random.randn(2, 2).astype(dtype) + 1j * np.random.randn(2, 2).astype(dtype), y=np.random.randn(3, 3).astype(dtype)))
        suite.addTest(ComplexKronTestCase(x=np.random.randn(2, 2).astype(dtype) + 1j * np.random.randn(2, 2).astype(dtype), y=np.random.randn(2, 2, 3).astype(dtype)))
    return suite
if __name__ == '__main__':
    unittest.main()