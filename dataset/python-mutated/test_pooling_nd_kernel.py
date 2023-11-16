import unittest
import chainer
from chainer.functions.pooling import pooling_nd_kernel
from chainer import testing
from chainer.testing import attr

@testing.parameterize(*testing.product({'ndim': [2, 3, 4]}))
@attr.gpu
class TestPoolingNDKernelMemo(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        chainer.backends.cuda.clear_memo()

    def test_pooling_nd_kernel_forward_memo(self):
        if False:
            while True:
                i = 10
        ndim = self.ndim
        with testing.patch('chainer.functions.pooling.pooling_nd_kernel.PoolingNDKernelForward._generate', wraps=None) as m:
            pooling_nd_kernel.PoolingNDKernelForward.generate(ndim)
            m.assert_called_once_with(ndim)
            pooling_nd_kernel.PoolingNDKernelForward.generate(ndim)
            m.assert_called_once_with(ndim)

    def test_pooling_nd_kernel_backward_memo(self):
        if False:
            i = 10
            return i + 15
        ndim = self.ndim
        with testing.patch('chainer.functions.pooling.pooling_nd_kernel.PoolingNDKernelBackward._generate', wraps=None) as m:
            pooling_nd_kernel.PoolingNDKernelBackward.generate(ndim)
            m.assert_called_once_with(ndim)
            pooling_nd_kernel.PoolingNDKernelBackward.generate(ndim)
            m.assert_called_once_with(ndim)
testing.run_module(__name__, __file__)