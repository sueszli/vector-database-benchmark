import unittest
import numpy as np
import paddle
from paddle import base
np_minor_version = int(np.__version__.split('.')[1])

def numpy_corr(np_arr, rowvar=True, dtype='float64'):
    if False:
        while True:
            i = 10
    if np_minor_version < 20:
        return np.corrcoef(np_arr, rowvar=rowvar)
    return np.corrcoef(np_arr, rowvar=rowvar, dtype=dtype)

class Corr_Test(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.shape = [4, 5]

    def test_tensor_corr_default(self):
        if False:
            i = 10
            return i + 15
        typelist = ['float64', 'float32']
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for (idx, p) in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                np_arr = np.random.rand(*self.shape).astype(dtype)
                tensor = paddle.to_tensor(np_arr, place=p)
                corr = paddle.linalg.corrcoef(tensor)
                np_corr = numpy_corr(np_arr, rowvar=True, dtype=dtype)
                if dtype == 'float32':
                    np.testing.assert_allclose(np_corr, corr.numpy(), rtol=1e-05, atol=1e-05)
                else:
                    np.testing.assert_allclose(np_corr, corr.numpy(), rtol=1e-05)

    def test_tensor_corr_rowvar(self):
        if False:
            i = 10
            return i + 15
        typelist = ['float64', 'float32']
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for (idx, p) in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                np_arr = np.random.rand(*self.shape).astype(dtype)
                tensor = paddle.to_tensor(np_arr, place=p)
                corr = paddle.linalg.corrcoef(tensor, rowvar=False)
                np_corr = numpy_corr(np_arr, rowvar=False, dtype=dtype)
                if dtype == 'float32':
                    np.testing.assert_allclose(np_corr, corr.numpy(), rtol=1e-05, atol=1e-05)
                else:
                    np.testing.assert_allclose(np_corr, corr.numpy(), rtol=1e-05)

class Corr_Test2(Corr_Test):

    def setUp(self):
        if False:
            print('Hello World!')
        self.shape = [10]

class Corr_Test3(Corr_Test):

    def setUp(self):
        if False:
            print('Hello World!')
        self.shape = [4, 5]

class Corr_Test4(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.shape = [2, 5, 2]

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')

        def test_err():
            if False:
                for i in range(10):
                    print('nop')
            np_arr = np.random.rand(*self.shape).astype('float64')
            tensor = paddle.to_tensor(np_arr)
            covrr = paddle.linalg.corrcoef(tensor)
        self.assertRaises(ValueError, test_err)

class Corr_Comeplex_Test(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.dtype = 'complex128'

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        x1 = paddle.static.data(name=self.dtype, shape=[2], dtype=self.dtype)
        self.assertRaises(TypeError, paddle.linalg.corrcoef, x=x1)
        paddle.disable_static()

class Corr_Test5(Corr_Comeplex_Test):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = 'complex64'
if __name__ == '__main__':
    unittest.main()