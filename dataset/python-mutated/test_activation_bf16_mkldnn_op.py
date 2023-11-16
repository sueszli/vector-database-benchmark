import abc
import unittest
import numpy as np
from op_test import OpTestTool, convert_float_to_uint16
from scipy.special import erf
from test_activation_op import TestActivation
from test_gelu_op import gelu
from paddle.base import core

@OpTestTool.skip_if_not_cpu_bf16()
class MKLDNNBF16ActivationOp(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def config(self):
        if False:
            print('Hello World!')
        pass

    @abc.abstractmethod
    def op_forward(self, x):
        if False:
            return 10
        pass

    @abc.abstractmethod
    def op_grad(self, dout, x):
        if False:
            while True:
                i = 10
        pass

    def set_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'use_mkldnn': True}

    def init_data(self):
        if False:
            while True:
                i = 10
        self.x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype(np.float32)

    def setUp(self):
        if False:
            return 10
        self.dtype = np.uint16
        self.init_data()
        self.config()
        self.set_attrs()
        self.out = self.op_forward(self.x)
        self.inputs = {'X': convert_float_to_uint16(self.x)}
        self.outputs = {'Out': self.out}

    def calculate_grads(self):
        if False:
            return 10
        self.dx = self.op_grad(self.out, self.x)

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.calculate_grads()
        self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out', user_defined_grads=[self.dx], user_defined_grad_outputs=[convert_float_to_uint16(self.out)])

class TestMKLDNNSigmoidBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'sigmoid'

    def op_forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return 1 / (1 + np.exp(-x))

    def op_grad(self, dout, x):
        if False:
            for i in range(10):
                print('nop')
        return dout * self.op_forward(x) * (1 - self.op_forward(x))

class TestMKLDNNSqrtBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            print('Hello World!')
        self.op_type = 'sqrt'

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.x = np.random.uniform(1, 2, [2, 4, 3, 5]).astype(np.float32)

    def op_forward(self, x):
        if False:
            i = 10
            return i + 15
        return np.sqrt(x)

    def op_grad(self, dout, x):
        if False:
            print('Hello World!')
        return dout / (2 * np.sqrt(x))

class TestMKLDNNGeluErfBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            print('Hello World!')
        self.op_type = 'gelu'

    def op_forward(self, x):
        if False:
            return 10
        return gelu(x, False)

    def op_grad(self, dout, x):
        if False:
            return 10
        return dout * (0.5 + 0.5 * erf(x / np.sqrt(2)) + x / np.sqrt(2 * np.pi) * np.exp(-0.5 * np.power(x, 2)))

class TestMKLDNNGeluErfDim2BF16Op(TestMKLDNNGeluErfBF16Op):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.uniform(-1, 1, [11, 17]).astype(np.float32)

class TestMKLDNNGeluTanhBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'gelu'

    def op_forward(self, x):
        if False:
            print('Hello World!')
        return gelu(x, True)

    def op_grad(self, dout, x):
        if False:
            return 10
        grad_part = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
        return dout * 0.5 * (1 + grad_part) * (1 + np.sqrt(2 / np.pi) * (x + 0.134145 * np.power(x, 3)) * (1 - grad_part))

    def set_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {'use_mkldnn': True, 'approximate': True}

class TestMKLDNNGeluTanhDim2BF16Op(TestMKLDNNGeluTanhBF16Op):

    def init_data(self):
        if False:
            return 10
        self.x = np.random.uniform(-1, 1, [11, 17]).astype(np.float32)

class TestMKLDNNReluBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'relu'

    def op_forward(self, x):
        if False:
            i = 10
            return i + 15
        return np.maximum(x, 0)

    def op_grad(self, dout, x):
        if False:
            for i in range(10):
                print('nop')
        return dout

class TestMKLDNNMishBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'mish'

    def op_forward(self, x):
        if False:
            print('Hello World!')
        return x * np.tanh(np.log(1 + np.exp(x)))

    def op_grad(self, dout, x):
        if False:
            i = 10
            return i + 15
        omega = np.exp(3 * x) + 4 * np.exp(2 * x) + np.exp(x) * (4 * x + 6) + 4 * (x + 1)
        delta = np.exp(2 * x) + 2 * np.exp(x) + 2
        return dout * (np.exp(x) * omega / delta ** 2)

class TestMKLDNNRelu6BF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            return 10
        self.op_type = 'relu6'

    def op_forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return np.clip(x, 0, 6)

    def op_grad(self, dout, x):
        if False:
            while True:
                i = 10
        return np.where((x > 0) & (x <= 6), dout, 0)

class TestMKLDNNLeakyReluBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'leaky_relu'

    def op_forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return np.where(x > 0, x, self.alpha * x)

    def op_grad(self, dout, x):
        if False:
            for i in range(10):
                print('nop')
        return np.where(x > 0, dout, self.alpha * dout)

    def set_attrs(self):
        if False:
            i = 10
            return i + 15
        self.alpha = 0.2
        self.attrs = {'use_mkldnn': True, 'alpha': self.alpha}

class TestMKLDNNSwishBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'swish'

    def expit(self, val):
        if False:
            while True:
                i = 10
        return 1 / (1 + np.exp(-self.beta * val))

    def op_forward(self, x):
        if False:
            return 10
        return x * self.expit(x)

    def op_grad(self, dout, x):
        if False:
            return 10
        return dout * self.expit(x) * (1 + self.beta * x * (1 - self.expit(x)))

    def set_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.beta = 0.2
        self.attrs = {'use_mkldnn': True, 'beta': self.beta}

class TestMKLDNNHardSwishBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'hard_swish'

    def op_forward(self, x):
        if False:
            print('Hello World!')
        result = np.where(x < -3, 0, x)
        return np.where(result > 3, result, result * (result + 3) / 6)

    def op_grad(self, dout, x):
        if False:
            return 10
        result = np.where(x < -3, 0, x)
        return np.where(result > 3, dout, dout * (2 * x + 3) / 6)

class TestMKLDNNTanhBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            while True:
                i = 10
        self.op_type = 'tanh'

    def op_forward(self, x):
        if False:
            i = 10
            return i + 15
        return np.tanh(x)

    def op_grad(self, dout, x):
        if False:
            print('Hello World!')
        return dout * (1 - np.tanh(x) ** 2)

class TestMKLDNNAbsBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'abs'

    def op_forward(self, x):
        if False:
            return 10
        return np.absolute(x)

    def op_grad(self, dout, x):
        if False:
            i = 10
            return i + 15
        return dout * np.sign(x)

class TestMKLDNNEluBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            print('Hello World!')
        self.op_type = 'elu'

    def op_forward(self, x):
        if False:
            print('Hello World!')
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def op_grad(self, dout, x):
        if False:
            print('Hello World!')
        return np.where(x > 0, dout, dout * self.alpha * np.exp(x))

    def set_attrs(self):
        if False:
            while True:
                i = 10
        self.alpha = 0.2
        self.attrs = {'use_mkldnn': True, 'alpha': self.alpha}

class TestMKLDNNExpBF16Op(MKLDNNBF16ActivationOp, TestActivation):

    def config(self):
        if False:
            return 10
        self.op_type = 'exp'

    def op_forward(self, x):
        if False:
            print('Hello World!')
        return np.exp(x)

    def op_grad(self, dout, x):
        if False:
            print('Hello World!')
        return dout * np.exp(x)
if __name__ == '__main__':
    unittest.main()