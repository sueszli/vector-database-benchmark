import unittest
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions.array.as_strided import _stride_array
from chainer import testing

def _broadcast_to(xp, x, shape):
    if False:
        print('Hello World!')
    if hasattr(xp, 'broadcast_to'):
        return xp.broadcast_to(x, shape)
    else:
        dummy = xp.empty(shape)
        return xp.broadcast_arrays(x, dummy)[0]

@testing.parameterize({'dtype': np.float16}, {'dtype': np.float32}, {'dtype': np.float64}, {'dtype': np.int16}, {'dtype': np.int32}, {'dtype': np.int64})
class TestStrideArray(unittest.TestCase):

    def check_flip(self, xp):
        if False:
            print('Hello World!')
        x = xp.arange(4, dtype=self.dtype)
        y = _stride_array(x, (4,), (-1,), 3)
        y_expected = x[::-1]
        testing.assert_allclose(y, y_expected)

    def test_flip_cpu(self):
        if False:
            return 10
        self.check_flip(np)

    @testing.attr.gpu
    def test_flip_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_flip(cuda.cupy)

    def check_broadcast(self, xp):
        if False:
            print('Hello World!')
        x = xp.arange(12, dtype=self.dtype).reshape((3, 4)).copy()
        y = _stride_array(x, (2, 3, 4), (0, 4, 1), 0)
        y_expected = _broadcast_to(xp, x, (2, 3, 4))
        testing.assert_allclose(y, y_expected)

    def test_broadcast_cpu(self):
        if False:
            i = 10
            return i + 15
        self.check_broadcast(np)

    @testing.attr.gpu
    def test_broadcast_gpu(self):
        if False:
            print('Hello World!')
        self.check_broadcast(cuda.cupy)

    def check_unstride(self, xp):
        if False:
            while True:
                i = 10
        x = xp.arange(12, dtype=self.dtype).reshape((3, 4))[::-1]
        y = _stride_array(x, (12,), (1,), 0)
        y_expected = xp.arange(12, dtype=self.dtype)
        testing.assert_allclose(y, y_expected)

    def test_unstride_cpu(self):
        if False:
            while True:
                i = 10
        self.check_unstride(np)

    @testing.attr.gpu
    def test_unstride_gpu(self):
        if False:
            print('Hello World!')
        self.check_unstride(cuda.cupy)

    def check_general_stride(self, xp):
        if False:
            for i in range(10):
                print('nop')
        x = xp.arange(8, dtype=self.dtype)
        y = _stride_array(x, (3, 3), (-1, 2), 3)
        y_expected = xp.array([[3, 5, 7], [2, 4, 6], [1, 3, 5]], dtype=self.dtype)
        testing.assert_allclose(y, y_expected)

    def test_general_stride_cpu(self):
        if False:
            return 10
        self.check_general_stride(np)

    @testing.attr.gpu
    def test_general_stride_gpu(self):
        if False:
            print('Hello World!')
        self.check_general_stride(cuda.cupy)

    def check_invalid_negative_index(self, xp):
        if False:
            for i in range(10):
                print('nop')
        x = xp.arange(8, dtype=self.dtype)
        with self.assertRaises(ValueError):
            _stride_array(x, (3, 3), (-1, 2), 1)

    def test_invalid_negative_index_cpu(self):
        if False:
            i = 10
            return i + 15
        self.check_invalid_negative_index(np)

    @testing.attr.gpu
    def test_invalid_negative_index_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_invalid_negative_index(cuda.cupy)

@testing.parameterize({'dtype': np.float16}, {'dtype': np.float32}, {'dtype': np.float64}, {'dtype': np.int16}, {'dtype': np.int32}, {'dtype': np.int64})
class TestAsStridedForward(unittest.TestCase):

    def check_flip_forward(self, xp):
        if False:
            while True:
                i = 10
        x = xp.arange(4, dtype=self.dtype)
        v = chainer.Variable(x)
        y = F.as_strided(v, (4,), (-1,), 3)
        y_expected = x[::-1]
        testing.assert_allclose(y.array, y_expected)

    def test_flip_forward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_flip_forward(np)

    @testing.attr.gpu
    def test_flip_forward_gpu(self):
        if False:
            print('Hello World!')
        self.check_flip_forward(cuda.cupy)

    def check_broadcast_forward(self, xp):
        if False:
            return 10
        x = xp.arange(12, dtype=self.dtype).reshape((3, 4)).copy()
        v = chainer.Variable(x)
        y = F.as_strided(v, (2, 3, 4), (0, 4, 1), 0)
        y_expected = _broadcast_to(xp, x, (2, 3, 4))
        testing.assert_allclose(y.array, y_expected)

    def test_broadcast_forward_cpu(self):
        if False:
            print('Hello World!')
        self.check_broadcast_forward(np)

    @testing.attr.gpu
    def test_broadcast_forward_gpu(self):
        if False:
            return 10
        self.check_broadcast_forward(cuda.cupy)

    def check_unstride_forward(self, xp):
        if False:
            while True:
                i = 10
        x = xp.arange(12, dtype=self.dtype).reshape((3, 4))[::-1]
        v = chainer.Variable(x)
        y = F.as_strided(v, (12,), (1,), 0)
        y_expected = xp.arange(12, dtype=self.dtype)
        testing.assert_allclose(y.array, y_expected)

    def test_unstride_forward_cpu(self):
        if False:
            print('Hello World!')
        self.check_unstride_forward(np)

    @testing.attr.gpu
    def test_unstride_forward_gpu(self):
        if False:
            i = 10
            return i + 15
        self.check_unstride_forward(cuda.cupy)

    def check_general_stride(self, xp):
        if False:
            while True:
                i = 10
        x = _stride_array(xp.arange(8, dtype=self.dtype), (3, 3), (-1, 2), 3)
        v = chainer.Variable(x)
        y = F.as_strided(v, (3, 3), (1, 2), 0)
        y_expected = _stride_array(xp.arange(8, dtype=self.dtype), (3, 3), (1, 2), 0)
        assert (y.array == y_expected).all()

    def test_general_stride_forward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_general_stride(np)

    @testing.attr.gpu
    def test_general_stride_forward_gpu(self):
        if False:
            return 10
        self.check_general_stride(cuda.cupy)

@testing.parameterize({'dtype': np.float16}, {'dtype': np.float32}, {'dtype': np.float64})
class TestAsStridedBackward(unittest.TestCase):

    def check_flip_backward(self, xp):
        if False:
            for i in range(10):
                print('nop')
        x = xp.arange(4, dtype=self.dtype)
        v = chainer.Variable(x)
        y = F.as_strided(v, (4,), (-1,), 3)
        y.grad = xp.ones((4,), dtype=self.dtype)
        (gx,) = chainer.grad((y,), (v,))
        testing.assert_allclose(gx.array, xp.ones((4,), dtype=self.dtype))

    def test_flip_backward_cpu(self):
        if False:
            while True:
                i = 10
        self.check_flip_backward(np)

    @testing.attr.gpu
    def test_flip_backward_gpu(self):
        if False:
            while True:
                i = 10
        self.check_flip_backward(cuda.cupy)

    def check_broadcast_backward(self, xp):
        if False:
            while True:
                i = 10
        x = xp.arange(12, dtype=self.dtype).reshape((3, 4)).copy()
        v = chainer.Variable(x)
        y = F.as_strided(v, (2, 3, 4), (0, 4, 1), 0)
        y.grad = xp.ones((2, 3, 4), dtype=self.dtype)
        (gx,) = chainer.grad((y,), (v,))
        testing.assert_allclose(gx.array, xp.ones(x.shape, dtype=self.dtype) * 2)

    def test_broadcast_backward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_broadcast_backward(np)

    @testing.attr.gpu
    def test_broadcast_backward_gpu(self):
        if False:
            return 10
        self.check_broadcast_backward(cuda.cupy)

    def check_unstride_backward(self, xp):
        if False:
            return 10
        x = xp.arange(12, dtype=self.dtype).reshape((3, 4))[::-1]
        v = chainer.Variable(x)
        y = F.as_strided(v, (12,), (1,), 0)
        y.grad = xp.ones((12,), dtype=self.dtype)
        (gx,) = chainer.grad((y,), (v,))
        testing.assert_allclose(gx.array, xp.ones(x.shape, dtype=self.dtype))

    def test_unstride_backward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_unstride_backward(np)

    @testing.attr.gpu
    def test_unstride_backward_gpu(self):
        if False:
            return 10
        self.check_unstride_backward(cuda.cupy)

    def check_general_stride_backward(self, xp):
        if False:
            while True:
                i = 10
        x = _stride_array(xp.arange(8, dtype=self.dtype), (3, 3), (-1, 2), 3)
        v = chainer.Variable(x)
        y = F.as_strided(v, (3, 3), (1, 2), 0)
        y.grad = xp.ones(y.shape, dtype=self.dtype)
        (gx,) = chainer.grad((y,), (v,))
        testing.assert_allclose(gx.array, xp.array([[0.5, 0.5, 0.0], [2.0, 2.0, 1.0], [1.0, 0.5, 0.5]], dtype=self.dtype))

    def test_general_stride_backward_cpu(self):
        if False:
            return 10
        self.check_general_stride_backward(np)

    @testing.attr.gpu
    def test_general_stride_backward_gpu(self):
        if False:
            print('Hello World!')
        self.check_general_stride_backward(cuda.cupy)

@testing.parameterize({'dtype': np.int16}, {'dtype': np.int32}, {'dtype': np.int64})
class TestAsStridedBackwardInvalidType(unittest.TestCase):

    def check_flip_backward(self, xp):
        if False:
            return 10
        x = xp.arange(4, dtype=self.dtype)
        v = chainer.Variable(x)
        y = F.as_strided(v, (4,), (-1,), 3)
        y.grad = xp.ones((4,), dtype=self.dtype)
        with self.assertRaises(TypeError):
            (gx,) = chainer.grad((y,), (v,))

    def test_flip_backward_cpu(self):
        if False:
            while True:
                i = 10
        self.check_flip_backward(np)

    @testing.attr.gpu
    def test_flip_backward_gpu(self):
        if False:
            while True:
                i = 10
        self.check_flip_backward(cuda.cupy)

    def check_broadcast_backward(self, xp):
        if False:
            while True:
                i = 10
        x = xp.arange(12, dtype=self.dtype).reshape((3, 4)).copy()
        v = chainer.Variable(x)
        y = F.as_strided(v, (2, 3, 4), (0, 4, 1), 0)
        y.grad = xp.ones((2, 3, 4), dtype=self.dtype)
        with self.assertRaises(TypeError):
            (gx,) = chainer.grad((y,), (v,))

    def test_broadcast_backward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_broadcast_backward(np)

    @testing.attr.gpu
    def test_broadcast_backward_gpu(self):
        if False:
            while True:
                i = 10
        self.check_broadcast_backward(cuda.cupy)

    def check_unstride_backward(self, xp):
        if False:
            i = 10
            return i + 15
        x = xp.arange(12, dtype=self.dtype).reshape((3, 4))[::-1]
        v = chainer.Variable(x)
        y = F.as_strided(v, (12,), (1,), 0)
        y.grad = xp.ones((12,), dtype=self.dtype)
        with self.assertRaises(TypeError):
            (gx,) = chainer.grad((y,), (v,))

    def test_unstride_backward_cpu(self):
        if False:
            print('Hello World!')
        self.check_unstride_backward(np)

    @testing.attr.gpu
    def test_unstride_backward_gpu(self):
        if False:
            i = 10
            return i + 15
        self.check_unstride_backward(cuda.cupy)

    def check_general_stride_backward(self, xp):
        if False:
            i = 10
            return i + 15
        x = _stride_array(xp.arange(8, dtype=self.dtype), (3, 3), (-1, 2), 3)
        v = chainer.Variable(x)
        y = F.as_strided(v, (3, 3), (1, 2), 0)
        y.grad = xp.ones(y.shape, dtype=self.dtype)
        with self.assertRaises(TypeError):
            (gx,) = chainer.grad((y,), (v,))

    def test_general_stride_backward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_general_stride_backward(np)

    @testing.attr.gpu
    def test_general_stride_backward_gpu(self):
        if False:
            while True:
                i = 10
        self.check_general_stride_backward(cuda.cupy)
testing.run_module(__name__, __file__)