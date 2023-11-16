import unittest
import numpy
import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer_tests.functions_tests.pooling_tests import pooling_nd_helper

def _pair(x):
    if False:
        i = 10
        return i + 15
    if isinstance(x, chainer.utils.collections_abc.Iterable):
        return x
    return (x, x)

@testing.parameterize(*testing.product({'sampling_ratio': [None, 1, 2, (None, 3), (1, 2), (numpy.int32(1), numpy.int32(2))], 'outsize': [5, 7, (5, 7), (numpy.int32(5), numpy.int32(7))], 'spatial_scale': [0.6, 1.0, 2.0, numpy.float32(0.6)]}))
class TestROIAlign2D(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        N = 3
        n_channels = 3
        self.x = pooling_nd_helper.shuffled_linspace((N, n_channels, 12, 8), numpy.float32)
        self.rois = numpy.array([[1, 1, 6, 6], [2, 6, 11, 7], [1, 3, 10, 5], [3, 3, 3, 3], [1.1, 2.2, 3.3, 4.4]], dtype=numpy.float32)
        self.roi_indices = numpy.array([0, 2, 1, 0, 2], dtype=numpy.int32)
        n_rois = self.rois.shape[0]
        outsize = _pair(self.outsize)
        self.gy = numpy.random.uniform(-1, 1, (n_rois, n_channels, outsize[0], outsize[1])).astype(numpy.float32)
        self.check_backward_options = {'atol': 0.0005, 'rtol': 0.005}

    def check_forward(self, x_data, roi_data, roi_index_data):
        if False:
            print('Hello World!')
        x = chainer.Variable(x_data)
        rois = chainer.Variable(roi_data)
        roi_indices = chainer.Variable(roi_index_data)
        y = functions.roi_average_align_2d(x, rois, roi_indices, outsize=self.outsize, spatial_scale=self.spatial_scale, sampling_ratio=self.sampling_ratio)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)
        self.assertEqual(self.gy.shape, y_data.shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        if False:
            i = 10
            return i + 15
        self.check_forward(self.x, self.rois, self.roi_indices)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        if False:
            i = 10
            return i + 15
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.rois), cuda.to_gpu(self.roi_indices))

    @attr.gpu
    @condition.retry(3)
    def test_forward_cpu_gpu_equal(self):
        if False:
            for i in range(10):
                print('nop')
        x_cpu = chainer.Variable(self.x)
        rois_cpu = chainer.Variable(self.rois)
        roi_indices_cpu = chainer.Variable(self.roi_indices)
        y_cpu = functions.roi_average_align_2d(x_cpu, rois_cpu, roi_indices_cpu, outsize=self.outsize, spatial_scale=self.spatial_scale, sampling_ratio=self.sampling_ratio)
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        rois_gpu = chainer.Variable(cuda.to_gpu(self.rois))
        roi_indices_gpu = chainer.Variable(cuda.to_gpu(self.roi_indices))
        y_gpu = functions.roi_average_align_2d(x_gpu, rois_gpu, roi_indices_gpu, outsize=self.outsize, spatial_scale=self.spatial_scale, sampling_ratio=self.sampling_ratio)
        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))

    def check_backward(self, x_data, roi_data, roi_index_data, y_grad):
        if False:
            print('Hello World!')

        def f(x, rois, roi_indices):
            if False:
                print('Hello World!')
            return functions.roi_average_align_2d(x, rois, roi_indices, outsize=self.outsize, spatial_scale=self.spatial_scale, sampling_ratio=self.sampling_ratio)
        gradient_check.check_backward(f, (x_data, roi_data, roi_index_data), y_grad, no_grads=[False, True, True], **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        if False:
            print('Hello World!')
        self.check_backward(self.x, self.rois, self.roi_indices, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        if False:
            return 10
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.rois), cuda.to_gpu(self.roi_indices), cuda.to_gpu(self.gy))
testing.run_module(__name__, __file__)