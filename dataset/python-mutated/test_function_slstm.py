import unittest
import numpy
import six
from chainer.backends import cuda
from chainer import functions
from chainer.functions.rnn import slstm
from chainer import gradient_check
from chainer import testing
from chainer.testing import backend

def _sigmoid(x):
    if False:
        i = 10
        return i + 15
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half

def inject_backend_tests(method_names):
    if False:
        while True:
            i = 10
    decorator = backend.inject_backend_tests(method_names, testing.product({'use_cuda': [False], 'use_ideep': ['never', 'always']}) + [{'use_cuda': True}])
    return decorator

@testing.parameterize(*testing.product_dict([{'shape': (5, 6, 2)}, {'shape': (8, 9, 4, 5)}, {'shape': (1, 0, 5)}], [{'dtype': numpy.float16}, {'dtype': numpy.float32}, {'dtype': numpy.float64}], [{'grad_outputs': (True, True)}, {'grad_outputs': (True, False)}, {'grad_outputs': (False, True)}], [{'flat': True}, {'flat': False}]))
@testing.fix_random()
@backend.inject_backend_tests(None, testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0']}) + testing.product({'use_cuda': [False], 'use_ideep': ['never', 'always']}) + testing.product([[{'use_cuda': True}], testing.product({'use_cudnn': ['never']}) + testing.product({'use_cudnn': ['always'], 'cudnn_deterministic': [True, False], 'autotune': [True, False]})]))
class TestSLSTM(testing.FunctionTestCase):
    dodge_nondifferentiable = True

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 0.001, 'rtol': 0.01}
            self.check_backward_options = {'atol': 0.005, 'rtol': 0.05}
            self.check_double_backward_options = {'atol': 0.005, 'rtol': 0.05}

    def generate_inputs(self):
        if False:
            return 10
        x_shape = []
        x_shape.append(self.shape[0])
        x_shape.append(4 * self.shape[1])
        for i in range(2, len(self.shape)):
            x_shape.append(self.shape[i])
        x_shape = tuple(x_shape)
        c1 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        c2 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        x1 = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        x2 = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        if self.flat:
            return (c1[..., 0], c2[..., 0], x1[..., 0], x2[..., 0])
        else:
            return (c1, c2, x1, x2)

    def forward(self, inputs, device):
        if False:
            print('Hello World!')
        (c1, c2, x1, x2) = inputs
        out = functions.slstm(c1, c2, x1, x2)
        return out

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        (c_prev1, c_prev2, x1, x2) = inputs

        def _extract_gates(x):
            if False:
                while True:
                    i = 10
            r = x.reshape((x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])
            return (r[:, :, i] for i in six.moves.range(4))
        (a1_in, i1_in, f1_in, o1_in) = _extract_gates(x1)
        (a2_in, i2_in, f2_in, o2_in) = _extract_gates(x2)
        c_expect = _sigmoid(i1_in) * numpy.tanh(a1_in) + _sigmoid(i2_in) * numpy.tanh(a2_in) + _sigmoid(f1_in) * c_prev1 + _sigmoid(f2_in) * c_prev2
        h_expect = _sigmoid(o1_in + o2_in) * numpy.tanh(c_expect)
        return (c_expect, h_expect)

    def generate_grad_outputs(self, outputs_template):
        if False:
            print('Hello World!')
        grad_out = []
        c = outputs_template[0]
        h = outputs_template[1]
        c_shape = c.shape
        h_shape = h.shape
        if self.grad_outputs[0] is True:
            grad_out.append(numpy.random.uniform(-1, 1, h_shape).astype(h.dtype))
        else:
            grad_out.append(None)
        if self.grad_outputs[1] is True:
            grad_out.append(numpy.random.uniform(-1, 1, c_shape).astype(c.dtype))
        else:
            grad_out.append(None)
        return tuple(grad_out)

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@testing.fix_random()
@inject_backend_tests(['test_backward'])
class TestSLSTMGrad(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        c_prev1 = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        c_prev2 = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        x1 = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(self.dtype)
        x2 = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(self.dtype)
        c_next = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        gc = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        gh = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        ggc_prev1 = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        ggc_prev2 = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        ggx1 = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(self.dtype)
        ggx2 = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(self.dtype)
        self.inputs = [c_prev1, c_prev2, x1, x2, c_next, gc, gh]
        self.grad_outputs = [ggc_prev1, ggc_prev2, ggx1, ggx2]
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_backward_options = {'dtype': numpy.float64, 'atol': 0.001, 'rtol': 0.01}

    def check_backward(self, inputs, grad_outputs, backend_config):
        if False:
            for i in range(10):
                print('nop')
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
        with backend_config:
            gradient_check.check_backward(slstm.SLSTMGrad(), inputs, grad_outputs, **self.check_backward_options)

    def test_backward(self, backend_config):
        if False:
            i = 10
            return i + 15
        self.check_backward(self.inputs, self.grad_outputs, backend_config)
testing.run_module(__name__, __file__)