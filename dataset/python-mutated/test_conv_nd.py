import itertools
import unittest
import numpy
from six import moves
from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr
from chainer.utils import conv_nd

class TestAsTuple(unittest.TestCase):

    def test_scalar(self):
        if False:
            for i in range(10):
                print('nop')
        actual = conv_nd.as_tuple(1, 3)
        expected = (1, 1, 1)
        self.assertEqual(actual, expected)

    def test_tuple(self):
        if False:
            while True:
                i = 10
        actual = conv_nd.as_tuple((1, 2, 3), 3)
        expected = (1, 2, 3)
        self.assertEqual(actual, expected)

    def test_list(self):
        if False:
            i = 10
            return i + 15
        actual = conv_nd.as_tuple([1, 2, 3], 3)
        expected = (1, 2, 3)
        self.assertEqual(actual, expected)

    def test_tuple_invalid_length(self):
        if False:
            return 10
        with self.assertRaises(AssertionError):
            conv_nd.as_tuple((1,), 3)

@testing.parameterize(*testing.product({'dims': [(10,), (10, 8), (10, 8, 6)]}))
class TestIm2ColND(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        shape = (2, 3) + self.dims
        self.img = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)

    def check_im2col_nd(self, ksize, stride, pad, gpu):
        if False:
            return 10
        dims = self.dims
        if gpu:
            img = cuda.to_gpu(self.img)
        else:
            img = self.img
        col = conv_nd.im2col_nd(img, ksize, stride, pad)
        outs = tuple((conv_nd.get_conv_outsize(d, k, s, p) for (d, k, s, p) in zip(dims, ksize, stride, pad)))
        expected_shape = (2, 3) + ksize + outs
        self.assertEqual(col.shape, expected_shape)
        col = cuda.to_cpu(col)
        for n in moves.range(2):
            for c in moves.range(3):
                for xs in itertools.product(*[moves.range(out) for out in outs]):
                    for dxs in itertools.product(*[moves.range(k) for k in ksize]):
                        oxs = tuple((x * s - p + dx for (x, s, p, dx) in zip(xs, stride, pad, dxs)))
                        if all((0 <= ox < d for (ox, d) in zip(oxs, dims))):
                            col_index = (n, c) + dxs + xs
                            img_index = (n, c) + oxs
                            self.assertEqual(col[col_index], self.img[img_index])
                        else:
                            col_index = (n, c) + dxs + xs
                            self.assertEqual(col[col_index], 0)

    def test_im2col_nd_1_cpu(self):
        if False:
            i = 10
            return i + 15
        ndim = len(self.dims)
        ksize = (1,) * ndim
        stride = (1,) * ndim
        pad = (1,) * ndim
        self.check_im2col_nd(ksize, stride, pad, gpu=False)

    def test_im2col_nd_2_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        ndim = len(self.dims)
        ksize = (2,) * ndim
        stride = (2,) * ndim
        pad = (2,) * ndim
        self.check_im2col_nd(ksize, stride, pad, gpu=False)

    def test_im2col_nd_3_cpu(self):
        if False:
            while True:
                i = 10
        ndim = len(self.dims)
        ksize = (1, 2, 1)[:ndim]
        stride = (2, 1, 2)[:ndim]
        pad = (1, 2, 1)[:ndim]
        self.check_im2col_nd(ksize, stride, pad, gpu=False)

    @attr.gpu
    def test_im2col_nd_1_gpu(self):
        if False:
            return 10
        ndim = len(self.dims)
        ksize = (1,) * ndim
        stride = (1,) * ndim
        pad = (1,) * ndim
        self.check_im2col_nd(ksize, stride, pad, gpu=True)

    @attr.gpu
    def test_im2col_nd_2_gpu(self):
        if False:
            print('Hello World!')
        ndim = len(self.dims)
        ksize = (2,) * ndim
        stride = (2,) * ndim
        pad = (2,) * ndim
        self.check_im2col_nd(ksize, stride, pad, gpu=True)

    @attr.gpu
    def test_im2col_nd_3_gpu(self):
        if False:
            while True:
                i = 10
        ndim = len(self.dims)
        ksize = (1, 2, 1)[:ndim]
        stride = (2, 1, 2)[:ndim]
        pad = (1, 2, 1)[:ndim]
        self.check_im2col_nd(ksize, stride, pad, gpu=True)

class TestIm2ColNDParameterRanks(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        shape = (2, 3, 4, 3)
        self.ksize = (2, 2)
        self.stride = (1, 1)
        self.pad = (0, 0)
        self.img = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)

    def test_im2col_nd_cpu_parameter_ranks(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(AssertionError):
            conv_nd.im2col_nd_cpu(self.img, (2,), self.stride, self.pad)
        with self.assertRaises(AssertionError):
            conv_nd.im2col_nd_cpu(self.img, self.ksize, (1,), self.pad)
        with self.assertRaises(AssertionError):
            conv_nd.im2col_nd_cpu(self.img, self.ksize, self.stride, (0,))

    @attr.gpu
    def test_im2col_nd_gpu_parameter_ranks(self):
        if False:
            print('Hello World!')
        img_gpu = cuda.to_gpu(self.img)
        with self.assertRaises(AssertionError):
            conv_nd.im2col_nd_gpu(img_gpu, (2,), self.stride, self.pad)
        with self.assertRaises(AssertionError):
            conv_nd.im2col_nd_gpu(img_gpu, self.ksize, (1,), self.pad)
        with self.assertRaises(AssertionError):
            conv_nd.im2col_nd_gpu(img_gpu, self.ksize, self.stride, (0,))

@testing.parameterize(*testing.product({'dims': [(10,), (10, 8), (10, 8, 6)]}))
class TestCol2ImND(unittest.TestCase):

    def check_col2im_nd(self, ksize, stride, pad, gpu):
        if False:
            print('Hello World!')
        dims = self.dims
        outs = tuple((conv_nd.get_conv_outsize(d, k, s, p) for (d, k, s, p) in zip(dims, ksize, stride, pad)))
        col_shape = (2, 3) + ksize + outs
        col = numpy.random.uniform(-1, 1, col_shape).astype(numpy.float32)
        if gpu:
            col_data = cuda.to_gpu(col)
        else:
            col_data = col
        img = conv_nd.col2im_nd(col_data, stride, pad, dims)
        img = cuda.to_cpu(img)
        img_shape = (2, 3) + dims
        self.assertEqual(img.shape, img_shape)
        for n in moves.range(2):
            for c in moves.range(3):
                for xs in itertools.product(*[moves.range(d) for d in dims]):
                    v = numpy.float32(0.0)
                    for dxs in itertools.product(*[moves.range(k) for k in ksize]):
                        oxs = tuple(((x + p - dx) // s for (x, p, dx, s) in zip(xs, pad, dxs, stride)))
                        if all(((x + p - dx) % s == 0 for (x, p, dx, s) in zip(xs, pad, dxs, stride))) and all((0 <= ox < out for (ox, out) in zip(oxs, outs))):
                            col_index = (n, c) + dxs + oxs
                            v += col[col_index]
                    img_index = (n, c) + xs
                    self.assertAlmostEqual(img[img_index], v)

    def test_col2im_1_cpu(self):
        if False:
            while True:
                i = 10
        ndim = len(self.dims)
        ksize = (1,) * ndim
        stride = (1,) * ndim
        pad = (1,) * ndim
        self.check_col2im_nd(ksize, stride, pad, gpu=False)

    def test_col2im_2_cpu(self):
        if False:
            i = 10
            return i + 15
        ndim = len(self.dims)
        ksize = (2,) * ndim
        stride = (2,) * ndim
        pad = (2,) * ndim
        self.check_col2im_nd(ksize, stride, pad, gpu=False)

    def test_col2im_3_cpu(self):
        if False:
            while True:
                i = 10
        ndim = len(self.dims)
        ksize = (1, 2, 1)[:ndim]
        stride = (2, 1, 2)[:ndim]
        pad = (1, 2, 1)[:ndim]
        self.check_col2im_nd(ksize, stride, pad, gpu=False)

    @attr.gpu
    def test_col2im_1_gpu(self):
        if False:
            print('Hello World!')
        ndim = len(self.dims)
        ksize = (1,) * ndim
        stride = (1,) * ndim
        pad = (1,) * ndim
        self.check_col2im_nd(ksize, stride, pad, gpu=True)

    @attr.gpu
    def test_col2im_2_gpu(self):
        if False:
            print('Hello World!')
        ndim = len(self.dims)
        ksize = (2,) * ndim
        stride = (2,) * ndim
        pad = (2,) * ndim
        self.check_col2im_nd(ksize, stride, pad, gpu=True)

    @attr.gpu
    def test_col2im_3_gpu(self):
        if False:
            print('Hello World!')
        ndim = len(self.dims)
        ksize = (1, 2, 1)[:ndim]
        stride = (2, 1, 2)[:ndim]
        pad = (1, 2, 1)[:ndim]
        self.check_col2im_nd(ksize, stride, pad, gpu=True)

class TestCol2ImNDParameterRanks(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.dims = (4, 3)
        self.ksize = (2, 2)
        self.stride = (1, 1)
        self.pad = (0, 0)
        self.outs = tuple((conv_nd.get_conv_outsize(d, k, s, p) for (d, k, s, p) in zip(self.dims, self.ksize, self.stride, self.pad)))
        col_shape = (2, 3) + self.ksize + self.outs
        self.col = numpy.random.uniform(-1, 1, col_shape).astype(numpy.float32)

    def test_col2im_nd_cpu_parameter_ranks(self):
        if False:
            i = 10
            return i + 15
        col_shape = (2, 3) + (2,) + self.outs
        col = numpy.random.uniform(-1, 1, col_shape).astype(numpy.float32)
        with self.assertRaises(AssertionError):
            conv_nd.col2im_nd_cpu(col, self.stride, self.pad, self.dims)
        with self.assertRaises(AssertionError):
            conv_nd.col2im_nd_cpu(self.col, (1,), self.pad, self.dims)
        with self.assertRaises(AssertionError):
            conv_nd.col2im_nd_cpu(self.col, self.stride, (0,), self.dims)
        with self.assertRaises(AssertionError):
            conv_nd.col2im_nd_cpu(self.col, self.stride, self.pad, (4,))

    @attr.gpu
    def test_col2im_nd_gpu_parameter_ranks(self):
        if False:
            for i in range(10):
                print('nop')
        col_shape = (2, 3) + (2,) + self.outs
        col = numpy.random.uniform(-1, 1, col_shape).astype(numpy.float32)
        col_gpu = cuda.to_gpu(col)
        with self.assertRaises(AssertionError):
            conv_nd.col2im_nd_gpu(col_gpu, self.stride, self.pad, self.dims)
        col_gpu = cuda.to_gpu(self.col)
        with self.assertRaises(AssertionError):
            conv_nd.col2im_nd_gpu(col_gpu, (1,), self.pad, self.dims)
        with self.assertRaises(AssertionError):
            conv_nd.col2im_nd_gpu(col_gpu, self.stride, (0,), self.dims)
        with self.assertRaises(AssertionError):
            conv_nd.col2im_nd_gpu(col_gpu, self.stride, self.pad, (4,))
testing.run_module(__name__, __file__)