import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle.pir_utils import test_with_pir_api

def valid_eigh_result(A, eigh_value, eigh_vector, uplo):
    if False:
        return 10
    assert A.ndim == 2 or A.ndim == 3
    if A.ndim == 2:
        valid_single_eigh_result(A, eigh_value, eigh_vector, uplo)
        return
    for (batch_A, batch_w, batch_v) in zip(A, eigh_value, eigh_vector):
        valid_single_eigh_result(batch_A, batch_w, batch_v, uplo)

def valid_single_eigh_result(A, eigh_value, eigh_vector, uplo):
    if False:
        while True:
            i = 10
    FP32_MAX_RELATIVE_ERR = 5e-05
    FP64_MAX_RELATIVE_ERR = 1e-14
    if A.dtype == np.single or A.dtype == np.csingle:
        rtol = FP32_MAX_RELATIVE_ERR
    else:
        rtol = FP64_MAX_RELATIVE_ERR
    (M, N) = A.shape
    triangular_func = np.tril if uplo == 'L' else np.triu
    if not np.iscomplexobj(A):
        A = triangular_func(A) + triangular_func(A, -1).T
    else:
        A = triangular_func(A) + np.matrix(triangular_func(A, -1)).H
    T = np.diag(eigh_value)
    residual = A - eigh_vector @ T @ np.linalg.inv(eigh_vector)
    np.testing.assert_array_less(np.linalg.norm(residual, np.inf) / (N * np.linalg.norm(A, np.inf)), rtol)
    residual = np.eye(M) - eigh_vector @ np.linalg.inv(eigh_vector)
    np.testing.assert_array_less(np.linalg.norm(residual, np.inf) / M, rtol)

class TestEighOp(OpTest):

    def setUp(self):
        if False:
            return 10
        paddle.enable_static()
        self.op_type = 'eigh'
        self.python_api = paddle.linalg.eigh
        self.init_input()
        self.init_config()
        np.random.seed(123)
        (out_w, out_v) = np.linalg.eigh(self.x_np, self.UPLO)
        self.inputs = {'X': self.x_np}
        self.attrs = {'UPLO': self.UPLO}
        self.outputs = {'Eigenvalues': out_w, 'Eigenvectors': out_v}

    def init_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.UPLO = 'L'

    def init_input(self):
        if False:
            while True:
                i = 10
        self.x_shape = (10, 10)
        self.x_type = np.float64
        self.x_np = np.random.random(self.x_shape).astype(self.x_type)

    def test_grad(self):
        if False:
            return 10
        self.check_grad(['X'], ['Eigenvalues'], check_pir=True)

class TestEighUPLOCase(TestEighOp):

    def init_config(self):
        if False:
            i = 10
            return i + 15
        self.UPLO = 'U'

class TestEighGPUCase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [32, 32]
        self.dtype = 'float32'
        self.UPLO = 'L'
        np.random.seed(123)
        self.x_np = np.random.random(self.x_shape).astype(self.dtype)

    def test_check_output_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        if paddle.is_compiled_with_cuda():
            paddle.disable_static(place=paddle.CUDAPlace(0))
            input_real_data = paddle.to_tensor(self.x_np)
            (actual_w, actual_v) = paddle.linalg.eigh(input_real_data, self.UPLO)
            valid_eigh_result(self.x_np, actual_w.numpy(), actual_v.numpy(), self.UPLO)

class TestEighAPI(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.init_input_data()
        self.UPLO = 'L'
        self.rtol = 1e-05
        self.atol = 1e-05
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
        np.random.seed(123)

    def init_input_shape(self):
        if False:
            print('Hello World!')
        self.x_shape = [5, 5]

    def init_input_data(self):
        if False:
            return 10
        self.init_input_shape()
        self.dtype = 'float32'
        self.real_data = np.random.random(self.x_shape).astype(self.dtype)
        complex_data = np.random.random(self.x_shape).astype(self.dtype) + 1j * np.random.random(self.x_shape).astype(self.dtype)
        self.trans_dims = list(range(len(self.x_shape) - 2)) + [len(self.x_shape) - 1, len(self.x_shape) - 2]
        self.complex_symm = np.divide(complex_data + np.conj(complex_data.transpose(self.trans_dims)), 2)

    def check_static_float_result(self):
        if False:
            i = 10
            return i + 15
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            input_x = paddle.static.data('input_x', shape=self.x_shape, dtype=self.dtype)
            (output_w, output_v) = paddle.linalg.eigh(input_x)
            exe = paddle.static.Executor(self.place)
            (actual_w, actual_v) = exe.run(main_prog, feed={'input_x': self.real_data}, fetch_list=[output_w, output_v])
            valid_eigh_result(self.real_data, actual_w, actual_v, self.UPLO)

    def check_static_complex_result(self):
        if False:
            i = 10
            return i + 15
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            x_dtype = np.complex64 if self.dtype == 'float32' else np.complex128
            input_x = paddle.static.data('input_x', shape=self.x_shape, dtype=x_dtype)
            (output_w, output_v) = paddle.linalg.eigh(input_x)
            exe = paddle.static.Executor(self.place)
            (actual_w, actual_v) = exe.run(main_prog, feed={'input_x': self.complex_symm}, fetch_list=[output_w, output_v])
            valid_eigh_result(self.complex_symm, actual_w, actual_v, self.UPLO)

    @test_with_pir_api
    def test_in_static_mode(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        self.check_static_float_result()
        self.check_static_complex_result()

    def test_in_dynamic_mode(self):
        if False:
            return 10
        paddle.disable_static()
        input_real_data = paddle.to_tensor(self.real_data)
        (actual_w, actual_v) = paddle.linalg.eigh(input_real_data)
        valid_eigh_result(self.real_data, actual_w.numpy(), actual_v.numpy(), self.UPLO)
        input_complex_data = paddle.to_tensor(self.complex_symm)
        (actual_w, actual_v) = paddle.linalg.eigh(input_complex_data)
        valid_eigh_result(self.complex_symm, actual_w.numpy(), actual_v.numpy(), self.UPLO)

    def test_eigh_grad(self):
        if False:
            return 10
        paddle.disable_static()
        x = paddle.to_tensor(self.complex_symm, stop_gradient=False)
        (w, v) = paddle.linalg.eigh(x)
        (w.sum() + paddle.abs(v).sum()).backward()
        np.testing.assert_allclose(abs(x.grad.numpy()), abs(x.grad.numpy().conj().transpose(self.trans_dims)), rtol=self.rtol, atol=self.atol)

class TestEighBatchAPI(TestEighAPI):

    def init_input_shape(self):
        if False:
            return 10
        self.x_shape = [2, 5, 5]

class TestEighAPIError(unittest.TestCase):

    def test_error(self):
        if False:
            while True:
                i = 10
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            input_x = paddle.static.data(name='x_1', shape=[12], dtype='float32')
            self.assertRaises(ValueError, paddle.linalg.eigh, input_x)
            input_x = paddle.static.data(name='x_2', shape=[12, 32], dtype='float32')
            self.assertRaises(ValueError, paddle.linalg.eigh, input_x)
            input_x = paddle.static.data(name='x_3', shape=[4, 4], dtype='float32')
            uplo = 'R'
            self.assertRaises(ValueError, paddle.linalg.eigh, input_x, uplo)
            input_x = paddle.static.data(name='x_4', shape=[4, 4], dtype='int32')
            self.assertRaises(TypeError, paddle.linalg.eigh, input_x)
if __name__ == '__main__':
    unittest.main()