import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import base
from paddle.base import Program, program_guard

class TestIdentityLossOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.max_relative_error = 0.006
        self.python_api = paddle.incubate.identity_loss
        self.inputs = {}
        self.initTestCase()
        self.dtype = np.float64
        self.op_type = 'identity_loss'
        self.attrs = {}
        self.attrs['reduction'] = self.reduction
        input = np.random.random(self.shape).astype(self.dtype)
        self.inputs['X'] = input
        if self.reduction == 0:
            output = input.sum()
        elif self.reduction == 1:
            output = input.mean()
        else:
            output = input
        self.outputs = {'Out': output}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        self.check_output()
        paddle.disable_static()

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        self.check_grad(['X'], 'Out')
        paddle.disable_static()

    def initTestCase(self):
        if False:
            i = 10
            return i + 15
        self.shape = (4, 10, 10)
        self.reduction = 0

class TestCase1(TestIdentityLossOp):

    def initTestCase(self):
        if False:
            i = 10
            return i + 15
        self.shape = (8, 16, 8)
        self.reduction = 0

class TestCase2(TestIdentityLossOp):

    def initTestCase(self):
        if False:
            print('Hello World!')
        self.shape = (8, 16)
        self.reduction = 1

class TestCase3(TestIdentityLossOp):

    def initTestCase(self):
        if False:
            print('Hello World!')
        self.shape = (4, 8, 16)
        self.reduction = 2

class TestIdentityLossFloat32(TestIdentityLossOp):

    def set_attrs(self):
        if False:
            return 10
        self.dtype = 'float32'

class TestIdentityLossOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            return 10
        paddle.enable_static()
        with program_guard(Program(), Program()):
            input_data = np.random.random((2, 4)).astype('float32')

            def test_int():
                if False:
                    i = 10
                    return i + 15
                paddle.incubate.identity_loss(x=input_data, reduction=3)
            self.assertRaises(Exception, test_int)

            def test_string():
                if False:
                    print('Hello World!')
                paddle.incubate.identity_loss(x=input_data, reduction='wrongkey')
            self.assertRaises(Exception, test_string)

            def test_dtype():
                if False:
                    i = 10
                    return i + 15
                x2 = paddle.static.data(name='x2', shape=[-1, 1], dtype='int32')
                paddle.incubate.identity_loss(x=x2, reduction=1)
            self.assertRaises(TypeError, test_dtype)
        paddle.disable_static()

class TestIdentityLossAPI(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.place = base.CPUPlace()

    def identity_loss_ref(self, input, reduction):
        if False:
            for i in range(10):
                print('nop')
        if reduction == 0 or reduction == 'sum':
            return input.sum()
        elif reduction == 1 or reduction == 'mean':
            return input.mean()
        else:
            return input

    def test_api_static(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_shape)
            out1 = paddle.incubate.identity_loss(x)
            out2 = paddle.incubate.identity_loss(x, reduction=0)
            out3 = paddle.incubate.identity_loss(x, reduction=1)
            out4 = paddle.incubate.identity_loss(x, reduction=2)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out1, out2, out3, out4])
        ref = [self.identity_loss_ref(self.x, 2), self.identity_loss_ref(self.x, 0), self.identity_loss_ref(self.x, 1), self.identity_loss_ref(self.x, 2)]
        for (out, out_ref) in zip(res, ref):
            np.testing.assert_allclose(out, out_ref, rtol=0.0001)

    def test_api_dygraph(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static(self.place)

        def test_case(x, reduction):
            if False:
                print('Hello World!')
            x_tensor = paddle.to_tensor(x)
            out = paddle.incubate.identity_loss(x_tensor, reduction)
            out_ref = self.identity_loss_ref(x, reduction)
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.0001)
        test_case(self.x, 0)
        test_case(self.x, 1)
        test_case(self.x, 2)
        test_case(self.x, 'sum')
        test_case(self.x, 'mean')
        test_case(self.x, 'none')
        paddle.enable_static()

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        x = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        x = paddle.to_tensor(x)
        self.assertRaises(Exception, paddle.incubate.identity_loss, x, -1)
        self.assertRaises(Exception, paddle.incubate.identity_loss, x, 3)
        self.assertRaises(Exception, paddle.incubate.identity_loss, x, 'wrongkey')
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [10, 12], 'int32')
            self.assertRaises(TypeError, paddle.incubate.identity_loss, x)
if __name__ == '__main__':
    unittest.main()