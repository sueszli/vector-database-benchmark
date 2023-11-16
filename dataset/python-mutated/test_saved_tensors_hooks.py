import unittest
import paddle
from paddle.autograd import PyLayer

class TestSavedTensorsHooks(unittest.TestCase):

    def test_save_for_multiply(self):
        if False:
            return 10

        def pack_hook(x):
            if False:
                while True:
                    i = 10
            return x.numpy()

        def unpack_hook(x):
            if False:
                return 10
            return paddle.to_tensor(x)
        a = paddle.ones([3, 3])
        b = paddle.ones([3, 3]) * 2
        a.stop_gradient = False
        b.stop_gradient = False
        with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
            y = paddle.multiply(a, b)
        y.sum().backward()
        aa = paddle.ones([3, 3])
        bb = paddle.ones([3, 3]) * 2
        aa.stop_gradient = False
        bb.stop_gradient = False
        yy = paddle.multiply(aa, bb)
        yy.sum().backward()
        self.assertTrue(paddle.equal_all(aa.grad, a.grad))
        self.assertTrue(paddle.equal_all(bb.grad, b.grad))

    def test_save_for_pylayer(self):
        if False:
            while True:
                i = 10

        class cus_multiply(PyLayer):

            @staticmethod
            def forward(ctx, a, b):
                if False:
                    i = 10
                    return i + 15
                y = paddle.multiply(a, b)
                ctx.save_for_backward(a, b)
                return y

            @staticmethod
            def backward(ctx, dy):
                if False:
                    for i in range(10):
                        print('nop')
                (a, b) = ctx.saved_tensor()
                grad_a = dy * a
                grad_b = dy * b
                return (grad_a, grad_b)

        def pack_hook(x):
            if False:
                return 10
            return x.numpy()

        def unpack_hook(x):
            if False:
                while True:
                    i = 10
            return paddle.to_tensor(x)
        a = paddle.ones([3, 3])
        b = paddle.ones([3, 3]) * 2
        a.stop_gradient = False
        b.stop_gradient = False
        with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
            y = cus_multiply.apply(a, b)
        y.sum().backward()
        aa = paddle.ones([3, 3])
        bb = paddle.ones([3, 3]) * 2
        aa.stop_gradient = False
        bb.stop_gradient = False
        yy = cus_multiply.apply(aa, bb)
        yy.sum().backward()
        self.assertTrue(paddle.equal_all(aa.grad, a.grad))
        self.assertTrue(paddle.equal_all(bb.grad, b.grad))
if __name__ == '__main__':
    unittest.main()