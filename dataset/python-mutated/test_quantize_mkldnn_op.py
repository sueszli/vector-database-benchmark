import unittest
import numpy as np
from op_test import OpTest
import paddle

class TestQuantizeOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'quantize'
        self.scale = 255.0
        self.shift = 0.0
        self.input_size = [1, 1, 5, 5]
        self.is_negative = False
        self.output_format = 'NCHW'
        self.set_scale()
        self.set_shift()
        self.set_is_negative()
        self.set_input_size()
        self.set_output_format()
        self.prepare_input()
        self.prepare_output()

    def prepare_input(self):
        if False:
            print('Hello World!')
        if self.is_negative:
            self.input = (2 * np.random.random_sample(self.input_size) - 1).astype('float32')
        else:
            self.input = np.random.random_sample(self.input_size).astype('float32')
        self.inputs = {'Input': OpTest.np_dtype_to_base_dtype(self.input)}
        self.attrs = {'Scale': self.scale, 'Shift': self.shift, 'is_negative_input': self.is_negative, 'output_format': self.output_format}

    def prepare_output(self):
        if False:
            return 10
        input_data_type = 'int8' if self.is_negative else 'uint8'
        output = np.rint(self.input * self.scale + self.shift).astype(input_data_type)
        self.outputs = {'Output': output}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)

    def check_raise_error(self, msg):
        if False:
            print('Hello World!')
        try:
            self.check_output()
        except Exception as e:
            if msg in str(e):
                raise AttributeError
            else:
                print(e)

    def set_scale(self):
        if False:
            while True:
                i = 10
        pass

    def set_shift(self):
        if False:
            while True:
                i = 10
        pass

    def set_is_negative(self):
        if False:
            i = 10
            return i + 15
        pass

    def set_input_size(self):
        if False:
            print('Hello World!')
        pass

    def set_output_format(self):
        if False:
            print('Hello World!')
        pass

class TestQuantizeOp1(TestQuantizeOp):

    def set_scale(self):
        if False:
            i = 10
            return i + 15
        self.scale = 127.0

    def set_is_negative(self):
        if False:
            print('Hello World!')
        self.is_nagative = True

class TestQuantizeOp2(TestQuantizeOp):

    def set_scale(self):
        if False:
            for i in range(10):
                print('nop')
        self.scale = 255.0

    def set_is_negative(self):
        if False:
            print('Hello World!')
        self.is_nagative = False

class TestQuantizeOpShift_NCHW_2_P(TestQuantizeOp):

    def set_output_format(self):
        if False:
            while True:
                i = 10
        self.output_format = 'NCHW'

    def set_is_negative(self):
        if False:
            i = 10
            return i + 15
        self.is_nagative = False

    def set_scale(self):
        if False:
            i = 10
            return i + 15
        self.scale = 255.0

    def set_shift(self):
        if False:
            print('Hello World!')
        self.shift = 0.0

    def set_input_size(self):
        if False:
            i = 10
            return i + 15
        self.input_size = [2, 3]

class TestQuantizeOpShift_NCHW_2_N(TestQuantizeOpShift_NCHW_2_P):

    def set_is_negative(self):
        if False:
            return 10
        self.is_nagative = True

    def set_scale(self):
        if False:
            i = 10
            return i + 15
        self.scale = 127.0

    def set_shift(self):
        if False:
            return 10
        self.shift = 128.0

class TestQuantizeOpShift_NHWC_2_P(TestQuantizeOpShift_NCHW_2_P):

    def set_output_format(self):
        if False:
            while True:
                i = 10
        self.output_format = 'NHWC'

class TestQuantizeOpShift_NHWC_2_N(TestQuantizeOpShift_NCHW_2_N):

    def set_output_format(self):
        if False:
            i = 10
            return i + 15
        self.output_format = 'NHWC'

class TestQuantizeOpShift_NCHW_3_P(TestQuantizeOpShift_NCHW_2_P):

    def set_input_size(self):
        if False:
            return 10
        self.input_size = [2, 3, 4]

class TestQuantizeOpShift_NCHW_3_N(TestQuantizeOpShift_NCHW_2_N):

    def set_input_size(self):
        if False:
            for i in range(10):
                print('nop')
        self.input_size = [2, 3, 4]

class TestQuantizeOpShift_NHWC_3_P(TestQuantizeOpShift_NCHW_3_P):

    def set_output_format(self):
        if False:
            print('Hello World!')
        self.output_format = 'NHWC'

class TestQuantizeOpShift_NHWC_3_N(TestQuantizeOpShift_NCHW_3_N):

    def set_output_format(self):
        if False:
            for i in range(10):
                print('nop')
        self.output_format = 'NHWC'

class TestQuantizeOpShift_NCHW_4_P(TestQuantizeOpShift_NCHW_2_P):

    def set_input_size(self):
        if False:
            print('Hello World!')
        self.input_size = [2, 3, 4, 5]

class TestQuantizeOpShift_NCHW_4_N(TestQuantizeOpShift_NCHW_2_N):

    def set_input_size(self):
        if False:
            i = 10
            return i + 15
        self.input_size = [2, 3, 4, 5]

class TestQuantizeOpShift_NHWC_4_P(TestQuantizeOpShift_NCHW_4_P):

    def set_output_format(self):
        if False:
            return 10
        self.output_format = 'NHWC'

class TestQuantizeOpShift_NHWC_4_N(TestQuantizeOpShift_NCHW_4_N):

    def set_output_format(self):
        if False:
            while True:
                i = 10
        self.output_format = 'NHWC'
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()