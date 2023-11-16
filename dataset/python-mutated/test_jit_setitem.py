import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase
import paddle
import paddle.nn.functional as F

class TestSetItemBase(Dy2StTestBase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        pass

    def init_data(self):
        if False:
            i = 10
            return i + 15
        paddle.seed(2023)
        x = paddle.randn([4, 8, 16, 32])
        x.stop_gradient = False
        return x

    def init_func(self):
        if False:
            i = 10
            return i + 15

        def foo(x):
            if False:
                return 10
            y = x + 1
            y[:, 2] = x[:, 2] + 99
            return y
        return foo

    def test_case(self):
        if False:
            return 10
        func = self.init_func()
        dy_res = self.run_dygraph(func)
        st_res = self.run_to_static(func)
        for (dy_out, st_out) in zip(dy_res, st_res):
            np.testing.assert_allclose(dy_out.numpy(), st_out.numpy())

    def run_dygraph(self, func):
        if False:
            for i in range(10):
                print('nop')
        x = self.init_data()
        y = func(x)
        x_grad = paddle.grad(y, x)[0]
        return (y, x_grad)

    def run_to_static(self, func):
        if False:
            return 10
        func = paddle.jit.to_static(func)
        return self.run_dygraph(func)

class TestCase1(TestSetItemBase):

    def init_func(self):
        if False:
            while True:
                i = 10

        def foo(x):
            if False:
                return 10
            y = x + 1
            y[2] = x[2] + 99
            return y
        return foo

class TestCase2(TestSetItemBase):

    def init_func(self):
        if False:
            i = 10
            return i + 15

        def foo(x):
            if False:
                return 10
            y = x + 1
            y[:] = x[:] + 99
            return y
        return foo

class TestCase3(TestSetItemBase):

    def init_func(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                i = 10
                return i + 15
            y = x + 1
            y[1::2] = x[1::2] + 99
            return y
        return foo

class TestCase4(TestSetItemBase):

    def init_func(self):
        if False:
            i = 10
            return i + 15

        def foo(x):
            if False:
                i = 10
                return i + 15
            y = x + 1
            y[1, 2] = x[1, 2] + 99
            return y
        return foo

class TestCase5(TestSetItemBase):

    def init_func(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                return 10
            y = x + 1
            y[[1, 2], [2, 3]] = x[[1, 2], [2, 3]] + 99
            return y
        return foo

class TestCase6(TestSetItemBase):

    def init_func(self):
        if False:
            while True:
                i = 10

        def foo(x):
            if False:
                return 10
            y = x + 1
            y[1, :, 3] = x[1, :, 3] + 99
            return y
        return foo

class TestCase7(TestSetItemBase):

    def init_func(self):
        if False:
            return 10

        def foo(x):
            if False:
                while True:
                    i = 10
            y = x + 1
            y[1, ..., 2] = x[1, ..., 2] + 99
            return y
        return foo

class TestCase8(TestSetItemBase):

    def init_func(self):
        if False:
            while True:
                i = 10

        def foo(x):
            if False:
                i = 10
                return i + 15
            y = x + 1
            index = paddle.to_tensor([1, 2], dtype='int64')
            y[index] = x[index] + 99
            return y
        return foo

class TestCase9(TestSetItemBase):

    def init_func(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                print('Hello World!')
            y = x + 1
            one = paddle.to_tensor(1, dtype='int64')
            two = paddle.to_tensor(2, dtype='int64')
            y[one, :, :, 2] = x[1, :, :, two] + 100
            return y
        return foo

class TestCase10(TestSetItemBase):

    def init_func(self):
        if False:
            i = 10
            return i + 15

        def foo(x):
            if False:
                print('Hello World!')
            y = x + 1
            y[..., 4:6] = y[..., 4:6] * 10000
            return y
        return foo

class TestCase11(TestSetItemBase):

    def init_func(self):
        if False:
            i = 10
            return i + 15

        def foo(x, value):
            if False:
                while True:
                    i = 10
            y = x + 1
            y[2, 4] = value
            return y
        return foo

    def run_dygraph(self, func):
        if False:
            for i in range(10):
                print('nop')
        x = self.init_data()
        value = paddle.ones((16, 32))
        value.stop_gradient = False
        y = func(x, value)
        (x_grad, value_grad) = paddle.grad(y, [x, value])
        return (y, x_grad, value_grad)

class TestCase12(TestSetItemBase):

    def init_func(self):
        if False:
            return 10

        def foo():
            if False:
                for i in range(10):
                    print('nop')
            res = paddle.zeros([4, 3, 2])
            b = paddle.zeros([4, 3, 2])
            v = paddle.to_tensor(1.0)
            for i in range(paddle.shape(b)[0]):
                res[i] = v
            return res
        return foo

    def run_dygraph(self, func):
        if False:
            while True:
                i = 10
        y = func()
        return (y,)

class TestCase13(TestSetItemBase):

    def init_func(self):
        if False:
            while True:
                i = 10

        def foo():
            if False:
                while True:
                    i = 10
            res = paddle.zeros([4, 3, 2])
            v = paddle.to_tensor(1.0)
            for i in range(4):
                res[i] = v
            return res
        return foo

    def run_dygraph(self, func):
        if False:
            i = 10
            return i + 15
        y = func()
        return (y,)

class TestCase14(TestSetItemBase):

    def init_func(self):
        if False:
            return 10

        def foo():
            if False:
                return 10
            data = np.arange(8).reshape((2, 4)).astype('float32')
            x = paddle.to_tensor(data)
            x[:, 1:] = x[:, :-1].clone()
            x[:, 0] = 1
            res = x.flatten()
            return res
        return foo

    def run_dygraph(self, func):
        if False:
            while True:
                i = 10
        y = func()
        return (y,)

class TestCase15(TestSetItemBase):

    def init_func(self):
        if False:
            return 10

        def foo(x, H, W):
            if False:
                while True:
                    i = 10
            (B, _, _, C) = x.shape
            pad_list = paddle.zeros([4], dtype='int32')
            pad_list[3] = H // 2
            pad_list[1] = W // 2
            x = F.pad(x, pad_list, data_format='NHWC')
            return x
        return foo

    def run_dygraph(self, func):
        if False:
            while True:
                i = 10
        x = paddle.ones((1, 6, 6, 3))
        H = paddle.full([1], 6, dtype='int32')
        W = paddle.full([1], 6, dtype='int32')
        y = func(x, H, W)
        return (y,)
if __name__ == '__main__':
    unittest.main()