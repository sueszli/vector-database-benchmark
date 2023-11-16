import unittest
import numpy as np
import paddle
from paddle.autograd.py_layer import PyLayer

class TestPyLayer(unittest.TestCase):

    def test_simple_pylayer_multiple_output(self):
        if False:
            for i in range(10):
                print('nop')

        class tanh(PyLayer):

            @staticmethod
            def forward(ctx, x1, x2, func1, func2=paddle.square):
                if False:
                    i = 10
                    return i + 15
                ctx.func = func2
                y1 = func1(x1)
                y2 = func1(x2)
                ctx.save_for_backward(y1, y2)
                return (y1, 1, y2, None)

            @staticmethod
            def backward(ctx, dy1, dy2):
                if False:
                    while True:
                        i = 10
                (y1, y2) = ctx.saved_tensor()
                re1 = dy1 * (1 - ctx.func(y1))
                re2 = dy2 * (1 - paddle.square(y2))
                return (re1, re2)
        input1 = paddle.randn([2, 3]).astype('float64')
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        z = tanh.apply(input1, input1, paddle.tanh, paddle.square)
        z = z[0] + z[2]
        z.mean().backward()
        z2 = paddle.tanh(input2) + paddle.tanh(input2)
        z2.mean().backward()
        self.assertTrue(np.max(np.abs(input1.grad.numpy() - input2.grad.numpy())) < 1e-10)

    def test_simple_pylayer_return_none_with_no_grad(self):
        if False:
            while True:
                i = 10

        class tanh(PyLayer):

            @staticmethod
            def forward(ctx, x1, x2, func1, func2=paddle.square):
                if False:
                    print('Hello World!')
                ctx.func = func2
                y1 = func1(x1)
                y2 = func1(x2)
                ctx.save_for_backward(y1, y2)
                return (1, None, y1, y2, '')

            @staticmethod
            def backward(ctx, dy1, dy2):
                if False:
                    for i in range(10):
                        print('nop')
                (y1, y2) = ctx.saved_tensor()
                re1 = dy1 * (1 - ctx.func(y1))
                re2 = dy2 * (1 - paddle.square(y2))
                return (re1, None)
        input1 = paddle.randn([2, 3]).astype('float64')
        input2 = input1.detach().clone()
        input3 = input1.detach().clone()
        input4 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        input3.stop_gradient = True
        input4.stop_gradient = True
        z = tanh.apply(input1, input3, paddle.tanh, paddle.square)
        z = z[2] + z[3]
        z.mean().backward()
        z2 = paddle.tanh(input2) + paddle.tanh(input4)
        z2.mean().backward()
        self.assertTrue(np.max(np.abs(input1.grad.numpy() - input2.grad.numpy())) < 1e-10)

    def test_simple_pylayer_single_output(self):
        if False:
            print('Hello World!')

        class tanh(PyLayer):

            @staticmethod
            def forward(ctx, x1, func1, func2=paddle.square):
                if False:
                    print('Hello World!')
                ctx.func = func2
                y1 = func1(x1)
                ctx.save_for_backward(y1)
                return y1

            @staticmethod
            def backward(ctx, dy1):
                if False:
                    for i in range(10):
                        print('nop')
                (y1,) = ctx.saved_tensor()
                re1 = dy1 * (1 - ctx.func(y1))
                return re1
        input1 = paddle.randn([2, 3]).astype('float64')
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        z = tanh.apply(x1=input1, func1=paddle.tanh)
        z.mean().backward()
        z2 = paddle.tanh(input2)
        z2.mean().backward()
        self.assertTrue(np.max(np.abs(input1.grad.numpy() - input2.grad.numpy())) < 1e-10)

    def test_simple_pylayer_multi_output(self):
        if False:
            for i in range(10):
                print('nop')

        class tanh(PyLayer):

            @staticmethod
            def forward(ctx, x1, func1, func2=paddle.split):
                if False:
                    while True:
                        i = 10
                ctx.func = func2
                y1 = func1(x1)
                ctx.save_for_backward(y1)
                return y1

            @staticmethod
            def backward(ctx, dy1):
                if False:
                    while True:
                        i = 10
                (y1,) = ctx.saved_tensor()
                re1 = ctx.func(dy1, 3)
                return re1
        input1 = paddle.randn([2, 3]).astype('float64')
        input2 = paddle.randn([2, 3]).astype('float64')
        input3 = paddle.randn([2, 3]).astype('float64')
        input1.stop_gradient = False
        input2.stop_gradient = False
        input3.stop_gradient = False
        z = tanh.apply(x1=[input1, input2, input3], func1=paddle.concat)
        z.mean().backward()
        z2 = paddle.concat([input1, input2, input3])
        z2.mean().backward()
        self.assertTrue(np.max(np.abs(input1.grad.numpy() - input2.grad.numpy())) < 1e-10)

    def test_pylayer_num_output_match(self):
        if False:
            print('Hello World!')

        class tanh(PyLayer):

            @staticmethod
            def forward(ctx, x1, x2):
                if False:
                    print('Hello World!')
                return x1 + x2

            @staticmethod
            def backward(ctx, dy1):
                if False:
                    i = 10
                    return i + 15
                return dy1 + 1
        input1 = paddle.randn([2, 3]).astype('float64')
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        z = tanh.apply(input1, input2)
        with self.assertRaises(ValueError):
            z.mean().backward()

    def test_pylayer_dtype(self):
        if False:
            for i in range(10):
                print('nop')

        class tanh(PyLayer):

            @staticmethod
            def forward(ctx, x, dtype):
                if False:
                    return 10
                y = paddle.cast(x, dtype)
                return y

            @staticmethod
            def backward(ctx, dy1):
                if False:
                    return 10
                return dy1
        dtypes = ['bool', 'float16', 'float32', 'float64', 'uint8', 'int32', 'int64']
        for dtype in dtypes:
            input1 = paddle.randn([2, 3])
            input1.stop_gradient = False
            self.assertIsNone(input1.grad)
            z = tanh.apply(input1, dtype)
            z = paddle.cast(z, 'float32')
            z.sum().backward()
            self.assertIsNotNone(input1.grad)

    def test_pylayer_Exception_forward(self):
        if False:
            for i in range(10):
                print('nop')

        class Layer_None1(PyLayer):

            @staticmethod
            def forward(ctx, *args):
                if False:
                    while True:
                        i = 10
                return None

            @staticmethod
            def backward(ctx, *args):
                if False:
                    return 10
                return args
        input1 = paddle.randn([2, 3]).astype('float64')
        with self.assertRaises(ValueError):
            z = Layer_None1.apply(input1)

        class Layer_None2(PyLayer):

            @staticmethod
            def forward(ctx, *args):
                if False:
                    for i in range(10):
                        print('nop')
                return [None, args[0]]

            @staticmethod
            def backward(ctx, *args):
                if False:
                    while True:
                        i = 10
                return args
        input1 = paddle.randn([2, 3]).astype('float64')
        z = Layer_None2.apply(input1)

        class Layer_one1(PyLayer):

            @staticmethod
            def forward(ctx, *args):
                if False:
                    while True:
                        i = 10
                return 1

            @staticmethod
            def backward(ctx, *args):
                if False:
                    for i in range(10):
                        print('nop')
                return args
        input1 = paddle.randn([2, 3]).astype('float64')
        with self.assertRaises(ValueError):
            z = Layer_one1.apply(input1)

        class Layer_one2(PyLayer):

            @staticmethod
            def forward(ctx, *args):
                if False:
                    while True:
                        i = 10
                return [1, 2, args[0]]

            @staticmethod
            def backward(ctx, *args):
                if False:
                    return 10
                return args
        input1 = paddle.randn([2, 3]).astype('float64')
        z = Layer_one2.apply(input1)

        class Layer_no_fw(PyLayer):

            @staticmethod
            def backward(ctx, *args):
                if False:
                    return 10
                return args
        input1 = paddle.randn([2, 3]).astype('float64')
        with self.assertRaises(NotImplementedError):
            z = Layer_no_fw.apply(input1)

    def test_pylayer_nograd(self):
        if False:
            while True:
                i = 10

        class tanh(PyLayer):

            @staticmethod
            def forward(ctx, x1, func1, func2=paddle.square, xx=None):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.func = func2
                y1 = func1(x1)
                return y1

            @staticmethod
            def backward(ctx, x1, y1, dy1):
                if False:
                    while True:
                        i = 10
                re1 = dy1 * (1 - ctx.func(y1))
                return re1
        input1 = paddle.randn([2, 3]).astype('float64')
        z = tanh.apply(input1, paddle.tanh, paddle.square)
        z.mean().backward()
        self.assertIsNone(z.grad)

    def test_pylayer_Exception_bk(self):
        if False:
            for i in range(10):
                print('nop')

        class Layer_bk_none1(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    print('Hello World!')
                return x * 2

            @staticmethod
            def backward(ctx, dy1):
                if False:
                    for i in range(10):
                        print('nop')
                return None
        input2 = paddle.randn([2, 3]).astype('float64')
        input2.stop_gradient = False
        z = Layer_bk_none1.apply(input2)
        z.sum().backward()
        self.assertEqual(input2.grad, None)

        class Layer_bk_none2(PyLayer):

            @staticmethod
            def forward(ctx, x1, x2):
                if False:
                    print('Hello World!')
                return x1 + x2

            @staticmethod
            def backward(ctx, dy1):
                if False:
                    i = 10
                    return i + 15
                return (None, dy1)
        input1 = paddle.randn([2, 3]).astype('float64')
        input1.stop_gradient = False
        z = Layer_bk_none2.apply(input1, input1)
        z.mean().backward()
        self.assertIsNone(z.grad)

        class Layer_bk_one1(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                return x + x

            @staticmethod
            def backward(ctx, dy):
                if False:
                    return 10
                return 1
        input1 = paddle.randn([2, 3]).astype('float64')
        input1.stop_gradient = False
        z = Layer_bk_one1.apply(input1)
        with self.assertRaises(ValueError):
            z.mean().backward()

        class Layer_bk_one2(PyLayer):

            @staticmethod
            def forward(ctx, x1, x2):
                if False:
                    for i in range(10):
                        print('nop')
                return (x1 * 2, x2 * 5)

            @staticmethod
            def backward(ctx, *args):
                if False:
                    return 10
                return (1, 1)
        input1 = paddle.randn([2, 3]).astype('float64')
        input1.stop_gradient = False
        y = Layer_bk_one2.apply(input1, input1)
        z = y[0] + y[1]
        with self.assertRaises(ValueError):
            z.mean().backward()

        class Layer_no_bk(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                return (x * 2, x * 5)
        input1 = paddle.randn([2, 3]).astype('float64')
        input1.stop_gradient = False
        z = Layer_no_bk.apply(input1)
        with self.assertRaises(OSError):
            z = z[0] + z[1]
            z.mean().backward()

        class Layer_bk_match(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    for i in range(10):
                        print('nop')
                return (x * 2, x * 5)

            @staticmethod
            def backward(ctx, dy1, dy2):
                if False:
                    return 10
                return (dy2 * 2, dy1 * 2)
        input1 = paddle.randn([2, 3]).astype('float64')
        input1.stop_gradient = False
        z = Layer_bk_match.apply(input1)
        with self.assertRaises(ValueError):
            z = z[0] + z[1]
            z.mean().backward()

    def test_pylayer_bk_return_none(self):
        if False:
            for i in range(10):
                print('nop')

        class Layer_bk_none1(PyLayer):

            @staticmethod
            def forward(ctx, x1, x2):
                if False:
                    return 10
                return x1 + x2

            @staticmethod
            def backward(ctx, dy):
                if False:
                    while True:
                        i = 10
                return 1
        input1 = paddle.randn([2, 3]).astype('float64')
        input2 = paddle.randn([2, 3]).astype('float64')
        input1.stop_gradient = True
        input2.stop_gradient = False
        z = Layer_bk_none1.apply(input1, input2)
        with self.assertRaises(ValueError):
            z.mean().backward()

        class Layer_bk_none2(PyLayer):

            @staticmethod
            def forward(ctx, x1, x2):
                if False:
                    while True:
                        i = 10
                return (x1 * 2, x2 * 5)

            @staticmethod
            def backward(ctx, *args):
                if False:
                    while True:
                        i = 10
                return (1, 1)
        input1 = paddle.randn([2, 3]).astype('float64')
        input2 = paddle.randn([2, 3]).astype('float64')
        input1.stop_gradient = True
        input2.stop_gradient = False
        z = Layer_bk_none2.apply(input1, input2)
        z = z[0] + z[1]
        with self.assertRaises(ValueError):
            z.mean().backward()

    def test_pylayer_inplace(self):
        if False:
            return 10

        class cus_tanh(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                return x

            @staticmethod
            def backward(ctx, dy):
                if False:
                    print('Hello World!')
                return dy

        class Layer(paddle.nn.Layer):

            def __init__(self):
                if False:
                    return 10
                super().__init__()

            def forward(self, data):
                if False:
                    for i in range(10):
                        print('nop')
                data = data ** 2
                z = paddle.tanh(data)
                z = cus_tanh.apply(data)
                return z.mean()
        for i in range(2):
            data = paddle.ones([2, 3], dtype='float64') / (i + 1)
            data.stop_gradient = False
            layer = Layer()
            z = layer(data)
            z.backward()
            self.assertIsNotNone(data.grad)

    def test_pylayer_inplace_backward_error(self):
        if False:
            return 10

        class cus_tanh(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    print('Hello World!')
                return x

            @staticmethod
            def backward(ctx, dy):
                if False:
                    i = 10
                    return i + 15
                return dy

        class Layer(paddle.nn.Layer):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()

            def forward(self, data):
                if False:
                    while True:
                        i = 10
                var_b = data ** 2
                var_c = var_b ** 2
                z = cus_tanh.apply(var_b)
                loss = paddle.nn.functional.relu(var_c)
                return loss
        data = paddle.ones([2, 3], dtype='float64')
        data.stop_gradient = False
        layer = Layer()
        z = layer(data)
        with self.assertRaisesRegex(RuntimeError, f'received tensor_version:{1} != wrapper_version_snapshot:{0}'):
            z.backward()

    def test_pylayer_inplace_backward_success_1(self):
        if False:
            return 10

        class cus_tanh(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                return x

            @staticmethod
            def backward(ctx, dy):
                if False:
                    return 10
                return dy

        class Layer(paddle.nn.Layer):

            def __init__(self):
                if False:
                    return 10
                super().__init__()

            def forward(self, data):
                if False:
                    for i in range(10):
                        print('nop')
                var_b = data ** 2
                var_c = cus_tanh.apply(var_b)
                var_d = var_c ** 2
                loss = var_d.sum()
                return loss
        for i in range(2):
            data = paddle.ones([2, 3], dtype='float64') / (i + 1)
            data.stop_gradient = False
            layer = Layer()
            z = layer(data)
            z.backward()
            self.assertIsNotNone(data.grad)

    def test_pylayer_inplace_backward_success_2(self):
        if False:
            i = 10
            return i + 15

        class cus_tanh(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                return x

            @staticmethod
            def backward(ctx, dy):
                if False:
                    i = 10
                    return i + 15
                return dy

        class Layer(paddle.nn.Layer):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()

            def forward(self, data):
                if False:
                    for i in range(10):
                        print('nop')
                var_b = data ** 2
                var_c = cus_tanh.apply(var_b)
                var_d = var_c + var_c
                loss = var_d.sum()
                return loss
        for i in range(2):
            data = paddle.ones([2, 3], dtype='float64') / (i + 1)
            data.stop_gradient = False
            layer = Layer()
            z = layer(data)
            z.backward()
            self.assertIsNotNone(data.grad)

    def test_pylayer_inplace_and_leaf_exception(self):
        if False:
            print('Hello World!')

        class cus_pylayer_op(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    print('Hello World!')
                return x

            @staticmethod
            def backward(ctx, dy):
                if False:
                    return 10
                return dy

        class Layer(paddle.nn.Layer):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()

            def forward(self, data):
                if False:
                    return 10
                z = cus_pylayer_op.apply(data)
                return z.mean()
        for i in range(2):
            data = paddle.ones([2, 3], dtype='float64') / (i + 1)
            data.stop_gradient = False
            layer = Layer()
            with self.assertRaises(ValueError):
                z = layer(data)

    def test_backward_in_backward(self):
        if False:
            i = 10
            return i + 15

        class cus_tanh(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    for i in range(10):
                        print('nop')
                temp = x.detach()
                ctx.inputs = temp
                return x.mean()

            @staticmethod
            def backward(ctx, dy):
                if False:
                    print('Hello World!')
                with paddle.set_grad_enabled(True):
                    temp = ctx.inputs
                    temp.stop_gradient = False
                    z = paddle.tanh(temp)
                    z.backward()
                    self.assertIsNotNone(temp.grad)
                    return paddle.to_tensor(temp.grad)
        for i in range(2):
            data = paddle.ones([2, 3], dtype='float32') / (i + 1)
            data.stop_gradient = False
            data = paddle.nn.functional.relu(data)
            z = paddle.tanh(data)
            z = cus_tanh.apply(data)

    def test_return_to_tensor(self):
        if False:
            i = 10
            return i + 15

        class Tanh(PyLayer):

            @staticmethod
            def forward(ctx, x1):
                if False:
                    while True:
                        i = 10
                y1 = paddle.tanh(x1)
                ctx.save_for_backward(y1)
                tensor_1 = paddle.to_tensor([1, 2], dtype='float32')
                return (y1, 5, None, 'helloworld', tensor_1)

            @staticmethod
            def backward(ctx, dy1, dy2):
                if False:
                    i = 10
                    return i + 15
                (y1,) = ctx.saved_tensor()
                re1 = dy1 * (1 - paddle.square(y1))
                return dy1
        input1 = paddle.randn([2, 3]).astype('float32')
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        (z, number, none_item, string_item, tensor1) = Tanh.apply(x1=input1)
        z.mean().backward()

    def test_materialize_grads(self):
        if False:
            print('Hello World!')

        class Tanh(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                ctx.mark_not_inplace(x)
                return (x, x + x)

            @staticmethod
            def backward(ctx, grad, grad2):
                if False:
                    for i in range(10):
                        print('nop')
                self.assertEqual(grad2, paddle.zeros([1]))
                return grad
        x = paddle.ones([1], dtype='float64')
        x.stop_gradient = False
        Tanh.apply(x)[0].backward()

    def test_dont_materialize_grads(self):
        if False:
            i = 10
            return i + 15

        class Tanh(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                ctx.mark_not_inplace(x)
                ctx.set_materialize_grads(False)
                return (x, x + x)

            @staticmethod
            def backward(ctx, grad, grad2):
                if False:
                    for i in range(10):
                        print('nop')
                self.assertIsNone(grad2)
                return grad
        x = paddle.ones([1], dtype='float64')
        x.stop_gradient = False
        Tanh.apply(x)[0].backward()

    def test_mark_non_differentiable(self):
        if False:
            while True:
                i = 10

        class Tanh(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    for i in range(10):
                        print('nop')
                a = x + x
                ctx.mark_non_differentiable(a)
                return a

            @staticmethod
            def backward(ctx, grad):
                if False:
                    while True:
                        i = 10
                self.assertTrue(False)
                return paddle.ones([1], dtype='float64')
        x = paddle.ones([1], dtype='float64')
        x.stop_gradient = False
        y = Tanh.apply(x)
        y.sum().backward()

    def test_mark_non_differentiable2(self):
        if False:
            while True:
                i = 10

        class Tanh(PyLayer):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                a = x + x
                b = x + x + x
                ctx.mark_non_differentiable(a)
                return (a, b)

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                if False:
                    return 10
                self.assertEqual(grad_a, paddle.zeros([1]))
                self.assertEqual(grad_b, paddle.ones([1], dtype='float64'))
                return grad_b
        x = paddle.ones([1], dtype='float64')
        x.stop_gradient = False
        (a, b) = Tanh.apply(x)
        b.sum().backward()
        self.assertEqual(x.grad, paddle.ones([1], dtype='float64'))
if __name__ == '__main__':
    unittest.main()