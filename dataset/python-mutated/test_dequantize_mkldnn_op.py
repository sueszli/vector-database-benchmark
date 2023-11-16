import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle

class TestDeQuantizeOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'dequantize'
        self.scale = 127.0
        self.shift = 0.0
        self.input_size = [1, 1, 5, 5]
        self.data_type = 'int8'
        self.set_scale()
        self.set_shift()
        self.set_data_type()
        self.set_input_size()
        if self.data_type == 'uint16':
            self.prepare_input_output_bf16()
        else:
            self.prepare_input_int8()
            self.prepare_output_int8()

    def prepare_input_output_bf16(self):
        if False:
            return 10
        output = np.random.random(self.input_size).astype(np.float32)
        input = convert_float_to_uint16(output)
        self.inputs = {'Input': OpTest.np_dtype_to_base_dtype(input)}
        self.outputs = {'Output': output}

    def prepare_input_int8(self):
        if False:
            print('Hello World!')
        if self.data_type == 'int8':
            self.input = (np.random.randint(0, 256, self.input_size) - 128).astype(self.data_type)
        else:
            self.input = np.random.randint(0, 256, self.input_size).astype(self.data_type)
        self.inputs = {'Input': OpTest.np_dtype_to_base_dtype(self.input)}
        self.attrs = {'Scale': self.scale, 'Shift': self.shift}

    def prepare_output_int8(self):
        if False:
            i = 10
            return i + 15
        output = (self.input / self.scale - self.shift / self.scale).astype('float')
        self.outputs = {'Output': output}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_dygraph=False)

    def check_raise_error(self, msg):
        if False:
            while True:
                i = 10
        try:
            self.check_output()
        except Exception as e:
            if msg in str(e):
                raise AttributeError
            else:
                print(e)

    def set_scale(self):
        if False:
            i = 10
            return i + 15
        pass

    def set_shift(self):
        if False:
            i = 10
            return i + 15
        pass

    def set_data_type(self):
        if False:
            while True:
                i = 10
        pass

    def set_input_size(self):
        if False:
            while True:
                i = 10
        pass

class TestDeQuantizeOp1(TestDeQuantizeOp):

    def set_scale(self):
        if False:
            for i in range(10):
                print('nop')
        self.scale = 1.5

    def set_data_type(self):
        if False:
            i = 10
            return i + 15
        self.data_type = 'int8'

class TestDeQuantizeOp2(TestDeQuantizeOp):

    def set_scale(self):
        if False:
            i = 10
            return i + 15
        self.scale = 0.8

    def set_data_type(self):
        if False:
            while True:
                i = 10
        self.data_type = 'uint8'

class TestDeQuantizeOpBf16(TestDeQuantizeOp):

    def set_scale(self):
        if False:
            i = 10
            return i + 15
        self.scale = 1.0

    def set_data_type(self):
        if False:
            while True:
                i = 10
        self.data_type = 'uint16'

class TestDeQuantizeOpShift_2_P(TestDeQuantizeOp):

    def set_data_type(self):
        if False:
            while True:
                i = 10
        self.data_type = 'uint8'

    def set_scale(self):
        if False:
            for i in range(10):
                print('nop')
        self.scale = 255.0

    def set_shift(self):
        if False:
            i = 10
            return i + 15
        self.shift = 128.0

    def set_input_size(self):
        if False:
            i = 10
            return i + 15
        self.input_size = [2, 3]

class TestDeQuantizeOpShift_2_N(TestDeQuantizeOpShift_2_P):

    def set_data_type(self):
        if False:
            while True:
                i = 10
        self.data_type = 'int8'

    def set_scale(self):
        if False:
            print('Hello World!')
        self.scale = 127.0

    def set_shift(self):
        if False:
            while True:
                i = 10
        self.shift = 10.0

    def set_input_size(self):
        if False:
            print('Hello World!')
        self.input_size = [2, 3]

class TestDeQuantizeOpShift_3_P(TestDeQuantizeOpShift_2_P):

    def set_input_size(self):
        if False:
            return 10
        self.input_size = [2, 3, 4]

class TestDeQuantizeOpShift_3_N(TestDeQuantizeOpShift_2_N):

    def set_input_size(self):
        if False:
            while True:
                i = 10
        self.input_size = [2, 3, 4]

class TestDeQuantizeOpShift_4_P(TestDeQuantizeOpShift_2_P):

    def set_input_size(self):
        if False:
            i = 10
            return i + 15
        self.input_size = [2, 3, 4, 5]

class TestDeQuantizeOpShift_4_N(TestDeQuantizeOpShift_2_N):

    def set_input_size(self):
        if False:
            return 10
        self.input_size = [2, 3, 4, 5]
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()