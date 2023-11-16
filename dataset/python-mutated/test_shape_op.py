import unittest
import numpy as np
from op import Operator
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle.base import core

class TestShapeOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'shape'
        self.python_api = paddle.shape
        self.config()
        input = np.zeros(self.shape, dtype=self.dtype)
        self.inputs = {'Input': input}
        self.outputs = {'Out': np.array(self.shape)}

    def config(self):
        if False:
            print('Hello World!')
        self.shape = [2, 3]
        self.dtype = np.float32

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_cinn=True, check_pir=True)

class case1(TestShapeOp):

    def config(self):
        if False:
            return 10
        self.shape = [2]
        self.dtype = np.float32

class case2(TestShapeOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = [1, 2, 3]
        self.dtype = np.float32

class TestShapeOpFp16(TestShapeOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = [2, 3]
        self.dtype = np.float16

class case1Fp16(TestShapeOp):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.shape = [2]
        self.dtype = np.float16

class case2Fp16(TestShapeOp):

    def config(self):
        if False:
            while True:
                i = 10
        self.shape = [1, 2, 3]
        self.dtype = np.float16

class TestShapeWithSelectedRows(unittest.TestCase):

    def get_places(self):
        if False:
            print('Hello World!')
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def check_with_place(self, place):
        if False:
            while True:
                i = 10
        scope = core.Scope()
        x_rows = [0, 1, 5, 4, 19]
        height = 20
        row_numel = 2
        np_array = np.ones((len(x_rows), row_numel)).astype('float32')
        x = scope.var('X').get_selected_rows()
        x.set_rows(x_rows)
        x.set_height(height)
        x_tensor = x.get_tensor()
        x_tensor.set(np_array, place)
        out_shape = scope.var('Out').get_tensor()
        op = Operator('shape', Input='X', Out='Out')
        op.run(scope, place)
        out_shape = np.array(out_shape).tolist()
        self.assertListEqual([5, 2], out_shape)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        for place in self.get_places():
            self.check_with_place(place)

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.supports_bfloat16(), 'core is not compiled with CUDA or place do not support bfloat16')
class TestShapeOpBf16(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'shape'
        self.dtype = 'bfloat16'
        self.python_api = paddle.shape
        self.config()
        input = np.zeros(self.shape)
        input = convert_float_to_uint16(input.astype('float32'))
        self.inputs = {'Input': input}
        self.outputs = {'Out': np.array(self.shape)}

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = [2, 3]

    def test_check_output(self):
        if False:
            return 10
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_cinn=True, check_pir=True)

class case1Bf16(TestShapeOpBf16):

    def config(self):
        if False:
            print('Hello World!')
        self.shape = [2]

class case2Bf16(TestShapeOpBf16):

    def config(self):
        if False:
            while True:
                i = 10
        self.shape = [1, 2, 3]
if __name__ == '__main__':
    unittest.main()