import unittest
import numpy
from chainer.backends import cuda
import chainer.functions as F
from chainer import initializers
from chainer.links.connection import deconvolution_nd
from chainer import testing
from chainer.testing import parameterize
from chainer.utils import conv

@parameterize(*testing.product({'dims': [(3, 2), (2,)], 'nobias': [True, False], 'dtype': [numpy.float32], 'used_outsize': ['case1', 'case2', 'None'], 'in_channels': [4, None, 'omit'], 'groups': [1, 2]}) + testing.product({'dims': [(4, 3, 2)], 'nobias': [False], 'dtype': [numpy.float16, numpy.float32, numpy.float64], 'used_outsize': ['None'], 'in_channels': [4, None, 'omit'], 'groups': [1, 2]}))
@testing.inject_backend_tests(['test_forward', 'test_backward'], [{}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}))
class TestDeconvolutionND(testing.LinkTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.N = 2
        self.out_channels = 2
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim
        if self.nobias:
            self.param_names = ('W',)
        else:
            self.param_names = ('W', 'b')
        if self.used_outsize == 'case1' or self.used_outsize == 'None':
            outs = tuple((conv.get_deconv_outsize(d, k, s, p) for (d, k, s, p) in zip(self.dims, self.ksize, self.stride, self.pad)))
        elif self.used_outsize == 'case2':
            outs = tuple((conv.get_deconv_outsize(d, k, s, p) + 1 for (d, k, s, p) in zip(self.dims, self.ksize, self.stride, self.pad)))
        if self.used_outsize != 'None':
            self.outsize = outs
        else:
            self.outsize = None
        self.x_shape = (self.N, 4) + self.dims
        self.check_backward_options.update({'eps': 0.01, 'atol': 0.0001, 'rtol': 0.001})
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 0.005, 'rtol': 0.05})
            self.check_backward_options.update({'eps': 2 ** (-3), 'atol': 0.01, 'rtol': 0.1})

    def before_test(self, test_name):
        if False:
            while True:
                i = 10
        using_old_cudnn = self.backend_config.xp is cuda.cupy and self.backend_config.use_cudnn == 'always' and (cuda.cuda.cudnn.getVersion() < 6000)
        if using_old_cudnn:
            self.check_backward_options.update({'eps': 2 ** (-3), 'atol': 0.01, 'rtol': 0.1})

    def generate_params(self):
        if False:
            return 10
        initial_bias = initializers.Uniform(scale=1.0, dtype=self.dtype)
        return (initial_bias,)

    def create_link(self, initializers):
        if False:
            while True:
                i = 10
        (initial_bias,) = initializers
        if self.in_channels == 'omit':
            link = deconvolution_nd.DeconvolutionND(self.ndim, self.out_channels, self.ksize, stride=self.stride, pad=self.pad, outsize=self.outsize, initial_bias=initial_bias, nobias=self.nobias, groups=self.groups)
        else:
            link = deconvolution_nd.DeconvolutionND(self.ndim, self.in_channels, self.out_channels, self.ksize, stride=self.stride, pad=self.pad, outsize=self.outsize, initial_bias=initial_bias, nobias=self.nobias, groups=self.groups)
        return link

    def generate_inputs(self):
        if False:
            return 10
        x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        return (x,)

    def forward_expected(self, link, inputs):
        if False:
            return 10
        (x,) = inputs
        W = link.W
        b = link.b
        y = F.deconvolution_nd(x, W, b, outsize=self.outsize, stride=self.stride, pad=self.pad, groups=self.groups)
        return (y.array,)

class TestDeconvolutionNDNoInitialBias(unittest.TestCase):

    def test_no_initial_bias(self):
        if False:
            return 10
        ndim = 3
        ksize = 3
        link = deconvolution_nd.DeconvolutionND(ndim, 3, 2, ksize, nobias=True)
        self.assertIsNone(link.b)

class TestDeconvolutionNDWrappers(unittest.TestCase):

    def _get_data(self, ndim):
        if False:
            print('Hello World!')
        in_channels = 3
        out_channels = 2
        dtype = numpy.float32
        x_shape = (2, in_channels) + (3,) * ndim
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        return (in_channels, out_channels, x)

    def test_deconv1d(self):
        if False:
            return 10
        (in_c, out_c, x) = self._get_data(1)
        link_nd = deconvolution_nd.DeconvolutionND(1, in_c, out_c, 2, initialW=1)
        link_1d = deconvolution_nd.Deconvolution1D(in_c, out_c, 2, initialW=1)
        testing.assert_allclose(link_nd(x).data, link_1d(x).data)

    def test_deconv3d(self):
        if False:
            print('Hello World!')
        (in_c, out_c, x) = self._get_data(3)
        link_nd = deconvolution_nd.DeconvolutionND(3, in_c, out_c, 2, initialW=1)
        link_3d = deconvolution_nd.Deconvolution3D(in_c, out_c, 2, initialW=1)
        testing.assert_allclose(link_nd(x).data, link_3d(x).data)
testing.run_module(__name__, __file__)