import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci
import paddle
from paddle.base import core

def get_outputs(DOut, X, Y):
    if False:
        return 10
    DX = np.dot(DOut, Y.T)
    DY = np.dot(X.T, DOut)
    DBias = np.sum(DOut, axis=0)
    return (DX, DY, DBias)

@skip_check_grad_ci(reason='no grap op')
@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFuseGemmEpilogueGradOpDXYBiasFP16(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'fused_gemm_epilogue_grad'
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()
        self.inputs = {'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5, 'X': np.random.random((8, 4)).astype(self.dtype) - 0.5, 'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5}
        self.attrs = {'activation_grad': 'none'}
        (DX, DY, DBias) = get_outputs(self.inputs['DOut'], self.inputs['X'], self.inputs['Y'])
        self.outputs = {'DX': DX, 'DY': DY, 'DBias': DBias}

    def init_dtype_type(self):
        if False:
            print('Hello World!')
        self.dtype = np.float16
        self.atol = 0.001

    def test_check_output(self):
        if False:
            while True:
                i = 10
        if self.dtype == np.float16 and (not core.is_float16_supported(self.place)):
            return
        self.check_output_with_place(self.place, atol=self.atol, check_dygraph=False)

@skip_check_grad_ci(reason='no grap op')
@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFuseGemmEpilogueGradOpDXYBiasFP32(TestFuseGemmEpilogueGradOpDXYBiasFP16):

    def init_dtype_type(self):
        if False:
            return 10
        self.dtype = np.single
        self.atol = 1e-06

@skip_check_grad_ci(reason='no grap op')
@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFuseGemmEpilogueGradOpDXYBiasFP64(TestFuseGemmEpilogueGradOpDXYBiasFP16):

    def init_dtype_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.double
        self.atol = 1e-06

@skip_check_grad_ci(reason='no grap op')
@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFuseGemmEpilogueGradOpDYBiasFP16(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'fused_gemm_epilogue_grad'
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()
        self.inputs = {'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5, 'X': np.random.random((8, 4)).astype(self.dtype) - 0.5, 'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5}
        self.attrs = {'activation_grad': 'none'}
        (_, DY, DBias) = get_outputs(self.inputs['DOut'], self.inputs['X'], self.inputs['Y'])
        self.outputs = {'DY': DY, 'DBias': DBias}

    def init_dtype_type(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float16
        self.atol = 0.001

    def test_check_output(self):
        if False:
            return 10
        if self.dtype == np.float16 and (not core.is_float16_supported(self.place)):
            return
        self.check_output_with_place(self.place, atol=self.atol, check_dygraph=False)

@skip_check_grad_ci(reason='no grap op')
@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFuseGemmEpilogueGradOpDYBiasFP32(TestFuseGemmEpilogueGradOpDYBiasFP16):

    def init_dtype_type(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.single
        self.atol = 1e-06

@skip_check_grad_ci(reason='no grap op')
@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFuseGemmEpilogueGradOpDYBiasFP64(TestFuseGemmEpilogueGradOpDYBiasFP16):

    def init_dtype_type(self):
        if False:
            while True:
                i = 10
        self.dtype = np.double
        self.atol = 1e-06

@skip_check_grad_ci(reason='no grap op')
@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFuseGemmEpilogueGradOpDYFP16(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'fused_gemm_epilogue_grad'
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()
        self.inputs = {'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5, 'X': np.random.random((8, 4)).astype(self.dtype) - 0.5, 'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5}
        self.attrs = {'activation_grad': 'none'}
        (_, DY, _) = get_outputs(self.inputs['DOut'], self.inputs['X'], self.inputs['Y'])
        self.outputs = {'DY': DY}

    def init_dtype_type(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float16
        self.atol = 0.001

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dtype == np.float16 and (not core.is_float16_supported(self.place)):
            return
        self.check_output_with_place(self.place, atol=self.atol, check_dygraph=False)

@skip_check_grad_ci(reason='no grap op')
@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFuseGemmEpilogueGradOpDYFP32(TestFuseGemmEpilogueGradOpDYFP16):

    def init_dtype_type(self):
        if False:
            print('Hello World!')
        self.dtype = np.single
        self.atol = 1e-06

@skip_check_grad_ci(reason='no grap op')
@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFuseGemmEpilogueGradOpDYFP64(TestFuseGemmEpilogueGradOpDYFP16):

    def init_dtype_type(self):
        if False:
            while True:
                i = 10
        self.dtype = np.double
        self.atol = 1e-06

@skip_check_grad_ci(reason='no grap op')
@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFuseGemmEpilogueGradOpDXYFP16(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'fused_gemm_epilogue_grad'
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()
        self.inputs = {'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5, 'X': np.random.random((8, 4)).astype(self.dtype) - 0.5, 'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5}
        self.attrs = {'activation_grad': 'none'}
        (DX, DY, _) = get_outputs(self.inputs['DOut'], self.inputs['X'], self.inputs['Y'])
        self.outputs = {'DX': DX, 'DY': DY}

    def init_dtype_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float16
        self.atol = 0.001

    def test_check_output(self):
        if False:
            while True:
                i = 10
        if self.dtype == np.float16 and (not core.is_float16_supported(self.place)):
            return
        self.check_output_with_place(self.place, atol=self.atol, check_dygraph=False)

@skip_check_grad_ci(reason='no grap op')
@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFuseGemmEpilogueGradOpDXYFP32(TestFuseGemmEpilogueGradOpDXYFP16):

    def init_dtype_type(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.single
        self.atol = 1e-06

@skip_check_grad_ci(reason='no grap op')
@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFuseGemmEpilogueGradOpDXYFP64(TestFuseGemmEpilogueGradOpDXYFP16):

    def init_dtype_type(self):
        if False:
            print('Hello World!')
        self.dtype = np.double
        self.atol = 1e-06
if __name__ == '__main__':
    paddle.enable_static()
    np.random.seed(0)
    unittest.main()