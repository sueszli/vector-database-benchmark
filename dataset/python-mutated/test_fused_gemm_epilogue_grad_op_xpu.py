import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
from paddle.base import core

def get_outputs(DOut, X, Y):
    if False:
        print('Hello World!')
    DX = np.dot(DOut, Y.T)
    DY = np.dot(X.T, DOut)
    DBias = np.sum(DOut, axis=0)
    return (DX, DY, DBias)

class XPUTestFuseGemmGradOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'fused_gemm_epilogue_grad'
        self.use_dynamic_create_class = False

    class TestFuseGemmEpilogueGradOpDXYBias1(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            paddle.enable_static()
            self.op_type = 'fused_gemm_epilogue_grad'
            self.__class__.no_need_check_grad = True
            self.dtype = self.in_type
            self.init_data()

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.inputs = {'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5, 'X': np.random.random((8, 4)).astype(self.dtype) - 0.5, 'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5}
            self.attrs = {'activation_grad': 'none'}
            (DX, DY, DBias) = get_outputs(self.inputs['DOut'], self.inputs['X'], self.inputs['Y'])
            self.outputs = {'DX': DX, 'DY': DY, 'DBias': DBias}

        def test_check_output(self):
            if False:
                return 10
            self.atol = 0.0001
            if self.dtype == np.float16:
                self.atol = 0.001
            self.check_output_with_place(core.XPUPlace(0), atol=self.atol)

    class TestFuseGemmEpilogueGradOpDXYBias2(XPUOpTest):

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.inputs = {'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5, 'X': np.random.random((8, 4)).astype(self.dtype) - 0.5, 'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5}
            self.attrs = {'activation_grad': 'none'}
            (_, DY, DBias) = get_outputs(self.inputs['DOut'], self.inputs['X'], self.inputs['Y'])
            self.outputs = {'DY': DY, 'DBias': DBias}

    class TestFuseGemmEpilogueGradOpDXYBias3(XPUOpTest):

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.inputs = {'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5, 'X': np.random.random((8, 4)).astype(self.dtype) - 0.5, 'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5}
            self.attrs = {'activation_grad': 'none'}
            (_, DY, _) = get_outputs(self.inputs['DOut'], self.inputs['X'], self.inputs['Y'])
            self.outputs = {'DY': DY}

    class TestFuseGemmEpilogueGradOpDXYBias4(XPUOpTest):

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.inputs = {'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5, 'X': np.random.random((8, 4)).astype(self.dtype) - 0.5, 'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5}
            self.attrs = {'activation_grad': 'none'}
            (DX, DY, _) = get_outputs(self.inputs['DOut'], self.inputs['X'], self.inputs['Y'])
            self.outputs = {'DX': DX, 'DY': DY}
support_types = get_xpu_op_support_types('fused_gemm_epilogue_grad')
for stype in support_types:
    create_test_class(globals(), XPUTestFuseGemmGradOp, stype)
if __name__ == '__main__':
    paddle.enable_static()
    np.random.seed(0)
    unittest.main()