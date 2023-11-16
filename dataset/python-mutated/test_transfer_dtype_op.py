import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
import paddle
from paddle.base import core

class TestTransferDtypeOpFp32ToFp64(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.astype('float64')}
        self.attrs = {'out_dtype': int(core.VarDesc.VarType.FP64), 'in_dtype': int(core.VarDesc.VarType.FP32)}
        self.op_type = 'transfer_dtype'

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_dygraph=False)

class TestTransferDtypeOpFp16ToFp32(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float16')}
        self.outputs = {'Out': ipt.astype('float32')}
        self.attrs = {'out_dtype': int(core.VarDesc.VarType.FP32), 'in_dtype': int(core.VarDesc.VarType.FP16)}
        self.op_type = 'transfer_dtype'

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(atol=0.001, check_dygraph=False)

class TestTransferDtypeOpFp32ToFp16(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.astype('float16')}
        self.attrs = {'out_dtype': int(core.VarDesc.VarType.FP16), 'in_dtype': int(core.VarDesc.VarType.FP32)}
        self.op_type = 'transfer_dtype'

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(atol=0.001, check_dygraph=False)

class TestTransferDtypeOpBf16ToFp32(OpTest):

    def setUp(self):
        if False:
            return 10
        ipt = np.array(np.random.randint(10, size=[10, 10])).astype('uint16')
        self.inputs = {'X': ipt}
        self.outputs = {'Out': convert_uint16_to_float(ipt)}
        self.attrs = {'out_dtype': int(core.VarDesc.VarType.FP32), 'in_dtype': int(core.VarDesc.VarType.BF16)}
        self.op_type = 'transfer_dtype'

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_dygraph=False)

class TestTransferDtypeFp32ToBf16(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        ipt = np.random.random(size=[10, 10]).astype('float32')
        self.inputs = {'X': ipt}
        self.outputs = {'Out': convert_float_to_uint16(ipt)}
        self.attrs = {'out_dtype': int(core.VarDesc.VarType.BF16), 'in_dtype': int(core.VarDesc.VarType.FP32)}
        self.op_type = 'transfer_dtype'

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_dygraph=False)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()