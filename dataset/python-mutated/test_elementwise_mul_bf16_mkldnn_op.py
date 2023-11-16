import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
from paddle import enable_static
from paddle.base import core

@unittest.skipIf(not core.supports_bfloat16(), 'place does not support BF16 evaluation')
class TestElementwiseMulBf16MklDNNOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'elementwise_mul'
        self.use_mkldnn = True
        self.mkldnn_data_type = 'bfloat16'
        self.axis = -1
        self.generate_data()
        self.x_bf16 = convert_float_to_uint16(self.x)
        self.y_bf16 = convert_float_to_uint16(self.y)
        self.inputs = {'X': self.x_bf16, 'Y': self.y_bf16}
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': convert_float_to_uint16(self.out)}

    def generate_data(self):
        if False:
            while True:
                i = 10
        self.x = np.random.random(100).astype(np.float32)
        self.y = np.random.random(100).astype(np.float32)
        self.out = np.multiply(self.x, self.y)

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad_normal(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad_with_place(core.CPUPlace(), ['X', 'Y'], 'Out', check_dygraph=False, user_defined_grads=[np.multiply(self.x, self.y), np.multiply(self.x, self.x)], user_defined_grad_outputs=[self.x_bf16])

    def test_check_grad_ingore_x(self):
        if False:
            return 10
        self.check_grad_with_place(core.CPUPlace(), ['Y'], 'Out', check_dygraph=False, user_defined_grads=[np.multiply(self.y, self.x)], user_defined_grad_outputs=[self.y_bf16])

    def test_check_grad_ingore_y(self):
        if False:
            print('Hello World!')
        self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out', check_dygraph=False, user_defined_grads=[np.multiply(self.x, self.y)], user_defined_grad_outputs=[self.x_bf16])

class TestElementwiseMulBroadcastingBf16MklDNNOp(TestElementwiseMulBf16MklDNNOp):

    def generate_data(self):
        if False:
            return 10
        self.x = np.random.uniform(1, 2, [1, 2, 3, 100]).astype(np.float32)
        self.y = np.random.uniform(1, 2, [100]).astype(np.float32)
        self.out = np.multiply(self.x, self.y)

    def compute_reduced_gradients(self, out_grads):
        if False:
            return 10
        part_sum = np.add.reduceat(out_grads, [0], axis=0)
        part_sum = np.add.reduceat(part_sum, [0], axis=1)
        part_sum = np.add.reduceat(part_sum, [0], axis=2)
        return part_sum.flatten()

    def test_check_grad_normal(self):
        if False:
            print('Hello World!')
        pass

    def test_check_grad_ingore_x(self):
        if False:
            for i in range(10):
                print('nop')
        pass
if __name__ == '__main__':
    enable_static()
    unittest.main()