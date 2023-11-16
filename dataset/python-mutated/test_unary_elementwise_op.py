import unittest
import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
import paddle

@OpTestTool.skip_if(not is_compiled_with_cuda(), 'x86 test will be skipped due to timeout.')
class TestUnaryOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_case()

    def init_case(self):
        if False:
            print('Hello World!')
        self.inputs = {'x': self.random([32, 64], 'float32', -10.0, 10.0)}

    def paddle_func(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.abs(x)

    def cinn_func(self, builder, x):
        if False:
            i = 10
            return i + 15
        return builder.abs(x)

    def build_paddle_program(self, target):
        if False:
            return 10
        x = paddle.to_tensor(self.inputs['x'], stop_gradient=True)
        out = self.paddle_func(x)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        if False:
            return 10
        builder = NetBuilder('unary_elementwise_test')
        x = builder.create_input(self.nptype2cinntype(self.inputs['x'].dtype), self.inputs['x'].shape, 'x')
        out = self.cinn_func(builder, x)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs['x']], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        if False:
            while True:
                i = 10
        self.check_outputs_and_grads()

class TestSqrtOp(TestUnaryOp):

    def init_case(self):
        if False:
            i = 10
            return i + 15
        self.inputs = {'x': self.random([32, 64], 'float32', 1.0, 1000.0)}

    def paddle_func(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.sqrt(x)

    def cinn_func(self, builder, x):
        if False:
            while True:
                i = 10
        return builder.sqrt(x)

class TestSqrtOpFP64(TestSqrtOp):

    def init_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'x': self.random([32, 64], 'float64', 1.0, 1000.0)}

class TestReluOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            for i in range(10):
                print('nop')
        return paddle.nn.functional.relu(x)

    def cinn_func(self, builder, x):
        if False:
            return 10
        return builder.relu(x)

class TestSigmoidOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.nn.functional.sigmoid(x)

    def cinn_func(self, builder, x):
        if False:
            print('Hello World!')
        return builder.sigmoid(x)

class TestIdentityOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            return 10
        return paddle.assign(x)

    def cinn_func(self, builder, x):
        if False:
            for i in range(10):
                print('nop')
        return builder.identity(x)

class TestExpOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.exp(x)

    def cinn_func(self, builder, x):
        if False:
            i = 10
            return i + 15
        return builder.exp(x)

class TestExpOpFP64(TestExpOp):

    def init_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'x': self.random([32, 64], 'float64', -10.0, 10.0)}

class TestErfOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            return 10
        return paddle.erf(x)

    def cinn_func(self, builder, x):
        if False:
            return 10
        return builder.erf(x)

class TestRsqrtOp(TestUnaryOp):

    def init_case(self):
        if False:
            return 10
        self.inputs = {'x': self.random([32, 64], 'float32', 1e-05, 1.0)}

    def paddle_func(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.rsqrt(x)

    def cinn_func(self, builder, x):
        if False:
            while True:
                i = 10
        return builder.rsqrt(x)

class TestLogOp(TestUnaryOp):

    def init_case(self):
        if False:
            return 10
        self.inputs = {'x': self.random([32, 64], 'float32', 1.0, 10.0)}

    def paddle_func(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.log(x)

    def cinn_func(self, builder, x):
        if False:
            while True:
                i = 10
        return builder.log(x)

class TestLog2Op(TestUnaryOp):

    def init_case(self):
        if False:
            i = 10
            return i + 15
        self.inputs = {'x': self.random([32, 64], 'float32', 1.0, 10.0)}

    def paddle_func(self, x):
        if False:
            while True:
                i = 10
        return paddle.log2(x)

    def cinn_func(self, builder, x):
        if False:
            i = 10
            return i + 15
        return builder.log2(x)

class TestLog10Op(TestUnaryOp):

    def init_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'x': self.random([32, 64], 'float32', 1.0, 10.0)}

    def paddle_func(self, x):
        if False:
            while True:
                i = 10
        return paddle.log10(x)

    def cinn_func(self, builder, x):
        if False:
            for i in range(10):
                print('nop')
        return builder.log10(x)

class TestFloorOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            print('Hello World!')
        return paddle.floor(x)

    def cinn_func(self, builder, x):
        if False:
            return 10
        return builder.floor(x)

class TestCeilOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.ceil(x)

    def cinn_func(self, builder, x):
        if False:
            i = 10
            return i + 15
        return builder.ceil(x)

class TestRoundOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            while True:
                i = 10
        return paddle.round(x)

    def cinn_func(self, builder, x):
        if False:
            for i in range(10):
                print('nop')
        return builder.round(x)

class TestTruncOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            return 10
        return paddle.trunc(x)

    def cinn_func(self, builder, x):
        if False:
            for i in range(10):
                print('nop')
        return builder.trunc(x)

class TestSinOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            while True:
                i = 10
        return paddle.sin(x)

    def cinn_func(self, builder, x):
        if False:
            print('Hello World!')
        return builder.sin(x)

class TestCosOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.cos(x)

    def cinn_func(self, builder, x):
        if False:
            i = 10
            return i + 15
        return builder.cos(x)

class TestTanOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            for i in range(10):
                print('nop')
        return paddle.tan(x)

    def cinn_func(self, builder, x):
        if False:
            for i in range(10):
                print('nop')
        return builder.tan(x)

class TestSinhOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.sinh(x)

    def cinn_func(self, builder, x):
        if False:
            print('Hello World!')
        return builder.sinh(x)

class TestCoshOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            print('Hello World!')
        return paddle.cosh(x)

    def cinn_func(self, builder, x):
        if False:
            print('Hello World!')
        return builder.cosh(x)

class TestTanhOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            print('Hello World!')
        return paddle.tanh(x)

    def cinn_func(self, builder, x):
        if False:
            return 10
        return builder.tanh(x)

class TestAsinOp(TestUnaryOp):

    def init_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'x': self.random([32, 64], 'float32', -1.0, 1.0)}

    def paddle_func(self, x):
        if False:
            while True:
                i = 10
        return paddle.asin(x)

    def cinn_func(self, builder, x):
        if False:
            for i in range(10):
                print('nop')
        return builder.asin(x)

class TestAcosOp(TestUnaryOp):

    def init_case(self):
        if False:
            return 10
        self.inputs = {'x': self.random([32, 64], 'float32', -1.0, 1.0)}

    def paddle_func(self, x):
        if False:
            print('Hello World!')
        return paddle.acos(x)

    def cinn_func(self, builder, x):
        if False:
            i = 10
            return i + 15
        return builder.acos(x)

class TestAtanOp(TestUnaryOp):

    def init_case(self):
        if False:
            while True:
                i = 10
        self.inputs = {'x': self.random([32, 64], 'float32', -1.0, 1.0)}

    def paddle_func(self, x):
        if False:
            while True:
                i = 10
        return paddle.atan(x)

    def cinn_func(self, builder, x):
        if False:
            return 10
        return builder.atan(x)

class TestAsinhOp(TestUnaryOp):

    def init_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'x': self.random([32, 64], 'float32', -1.0, 1.0)}

    def paddle_func(self, x):
        if False:
            return 10
        return paddle.asinh(x)

    def cinn_func(self, builder, x):
        if False:
            print('Hello World!')
        return builder.asinh(x)

class TestAcoshOp(TestUnaryOp):

    def init_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'x': self.random([32, 64], 'float32', 1.0, 100.0)}

    def paddle_func(self, x):
        if False:
            while True:
                i = 10
        return paddle.acosh(x)

    def cinn_func(self, builder, x):
        if False:
            return 10
        return builder.acosh(x)

class TestAtanhOp(TestUnaryOp):

    def init_case(self):
        if False:
            i = 10
            return i + 15
        self.inputs = {'x': self.random([32, 64], 'float32', -1.0, 1.0)}

    def paddle_func(self, x):
        if False:
            for i in range(10):
                print('nop')
        return paddle.atanh(x)

    def cinn_func(self, builder, x):
        if False:
            print('Hello World!')
        return builder.atanh(x)

class TestLogicalNotOp(TestUnaryOp):

    def init_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'x': self.random([32, 64], 'bool')}

    def paddle_func(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.logical_not(x)

    def cinn_func(self, builder, x):
        if False:
            print('Hello World!')
        return builder.logical_not(x)

class TestBitwiseNotOp(TestUnaryOp):

    def init_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'x': self.random([32, 64], 'int32', 1, 10000)}

    def paddle_func(self, x):
        if False:
            while True:
                i = 10
        return paddle.bitwise_not(x)

    def cinn_func(self, builder, x):
        if False:
            i = 10
            return i + 15
        return builder.bitwise_not(x)

class TestSignOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            for i in range(10):
                print('nop')
        return paddle.sign(x)

    def cinn_func(self, builder, x):
        if False:
            while True:
                i = 10
        return builder.sign(x)

class TestAbsOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            return 10
        return paddle.abs(x)

    def cinn_func(self, builder, x):
        if False:
            return 10
        return builder.abs(x)

class TestIsNanOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.isnan(x)

    def cinn_func(self, builder, x):
        if False:
            i = 10
            return i + 15
        return builder.is_nan(x)

class TestIsNanCase1(TestIsNanOp):

    def init_case(self):
        if False:
            i = 10
            return i + 15
        self.inputs = {'x': self.random([32, 64])}
        self.inputs['x'][0] = [np.nan] * 64

class TestIsFiniteOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            while True:
                i = 10
        return paddle.isfinite(x)

    def cinn_func(self, builder, x):
        if False:
            return 10
        return builder.is_finite(x)

class TestIsFiniteCase1(TestIsFiniteOp):

    def init_case(self):
        if False:
            print('Hello World!')
        self.inputs = {'x': self.random([32, 64])}
        self.inputs['x'][0] = [np.inf] * 64

class TestIsInfOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.isinf(x)

    def cinn_func(self, builder, x):
        if False:
            while True:
                i = 10
        return builder.is_inf(x)

class TestIsInfCase1(TestIsInfOp):

    def init_case(self):
        if False:
            print('Hello World!')
        self.inputs = {'x': self.random([32, 64])}
        self.inputs['x'][0] = [np.inf] * 64

class TestNegOp(TestUnaryOp):

    def paddle_func(self, x):
        if False:
            print('Hello World!')
        return paddle.neg(x)

    def cinn_func(self, builder, x):
        if False:
            print('Hello World!')
        return builder.negative(x)

class TestNegCase1(TestNegOp):

    def init_case(self):
        if False:
            return 10
        self.inputs = {'x': self.random([32, 64], low=-1.0, high=1.0)}
if __name__ == '__main__':
    unittest.main()