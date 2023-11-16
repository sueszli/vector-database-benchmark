import unittest
from struct import pack, unpack
import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
import paddle

@OpTestTool.skip_if(not is_compiled_with_cuda(), 'x86 test will be skipped due to timeout.')
class TestBitcastConvertOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.init_case()

    def init_case(self):
        if False:
            print('Hello World!')
        data = np.random.random([3, 1]).astype(np.int32)
        packed = pack(data.size * 'i', *data.flatten())
        self.inputs = {'x': data}
        self.outputs = {'y': np.array(unpack('12B', packed), dtype='uint8').reshape((3, 1, 4)), 'output_type': 'uint8'}

    def build_paddle_program(self, target):
        if False:
            return 10
        y = paddle.to_tensor(self.outputs['y'], stop_gradient=False)
        self.paddle_outputs = [y]

    def build_cinn_program(self, target):
        if False:
            print('Hello World!')
        builder = NetBuilder('bitcast_convert')
        x = builder.create_input(self.nptype2cinntype(self.inputs['x'].dtype), self.inputs['x'].shape, 'x')
        out = builder.bitcast_convert(x, self.outputs['output_type'])
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs['x']], [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_outputs_and_grads()

class TestBitcastConvertCase1(TestBitcastConvertOp):

    def init_case(self):
        if False:
            return 10
        data = np.random.random([4, 2]).astype(np.int16)
        packed = pack(data.size * 'h', *data.flatten())
        self.inputs = {'x': data}
        self.outputs = {'y': np.array(unpack('4i', packed), dtype='int32').reshape(4), 'output_type': 'int32'}

class TestBitcastConvertCase2(TestBitcastConvertOp):

    def init_case(self):
        if False:
            return 10
        data = np.random.random([4, 3, 2]).astype(np.float32)
        packed = pack(data.size * 'f', *data.flatten())
        self.inputs = {'x': data}
        self.outputs = {'y': np.array(unpack('12d', packed), dtype='float64').reshape((4, 3)), 'output_type': 'float64'}

class TestBitcastConvertCase3(TestBitcastConvertOp):

    def init_case(self):
        if False:
            for i in range(10):
                print('nop')
        data = np.random.random([4, 3, 2]).astype(np.float32)
        packed = pack(data.size * 'f', *data.flatten())
        self.inputs = {'x': data}
        self.outputs = {'y': np.array(unpack('48H', packed), dtype='uint16').reshape((4, 3, 2, 2)), 'output_type': 'uint16'}
if __name__ == '__main__':
    unittest.main()