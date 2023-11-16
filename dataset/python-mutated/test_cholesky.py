import numpy
import chainer.functions as F
from chainer import testing

@testing.parameterize(*testing.product({'dtype': [numpy.float32, numpy.float64], 'shape': [(5, 5), (1, 1)]}))
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}))
class TestCholesky(testing.FunctionTestCase):

    def random_matrix(self, shape, dtype, scale, sym=False):
        if False:
            for i in range(10):
                print('nop')
        (m, n) = shape[-2:]
        dtype = numpy.dtype(dtype)
        assert dtype.kind in 'iufc'
        (low_s, high_s) = scale
        bias = None
        if dtype.kind in 'iu':
            err = numpy.sqrt(m * n) / 2.0
            low_s += err
            high_s -= err
            if dtype.kind in 'u':
                assert sym, 'generating nonsymmetric matrix with uint cells is not supported'
                high_s = bias = high_s / (1 + numpy.sqrt(m * n))
        assert low_s <= high_s
        a = numpy.random.standard_normal(shape)
        (u, s, vh) = numpy.linalg.svd(a)
        new_s = numpy.random.uniform(low_s, high_s, s.shape)
        if sym:
            assert m == n
            new_a = numpy.einsum('...ij,...j,...kj', u, new_s, u)
        else:
            new_a = numpy.einsum('...ij,...j,...jk', u, new_s, vh)
        if bias is not None:
            new_a += bias
        if dtype.kind in 'iu':
            new_a = numpy.rint(new_a)
        return new_a.astype(dtype)

    def setUp(self):
        if False:
            while True:
                i = 10
        self.check_forward_options = {'atol': 0.001, 'rtol': 0.001}
        self.check_backward_options = {'atol': 0.001, 'rtol': 0.001, 'eps': 0.0001}
        self.check_double_backward_options = {'atol': 0.001, 'rtol': 0.001, 'eps': 0.0001}

    def generate_inputs(self):
        if False:
            while True:
                i = 10
        a = self.random_matrix(self.shape, self.dtype, scale=(0.01, 2.0), sym=True)
        return (a,)

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        (a,) = inputs
        a = 0.5 * (a + a.T)
        y_expect = numpy.linalg.cholesky(a)
        return (y_expect.astype(self.dtype),)

    def forward(self, inputs, device):
        if False:
            for i in range(10):
                print('nop')
        (a,) = inputs
        a = 0.5 * (a + a.T)
        y = F.cholesky(a)
        return (y,)
testing.run_module(__name__, __file__)