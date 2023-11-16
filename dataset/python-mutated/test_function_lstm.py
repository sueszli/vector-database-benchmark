import unittest
import numpy
import six
from chainer.backends import cuda
import chainer.functions as F
from chainer.functions.rnn import lstm
from chainer import gradient_check
from chainer import testing
from chainer.testing import backend

def sigmoid(x):
    if False:
        for i in range(10):
            print('nop')
    return numpy.tanh(x * 0.5) * 0.5 + 0.5

def _shaped_random(shape, dtype):
    if False:
        while True:
            i = 10
    return numpy.random.uniform(-1, 1, shape).astype(dtype)

def inject_backend_tests(method_names):
    if False:
        while True:
            i = 10
    decorator = backend.inject_backend_tests(method_names, testing.product({'use_cuda': [False], 'use_ideep': ['never', 'always']}) + [{'use_cuda': True}])
    return decorator

class LSTMTestBase(object):
    dodge_nondifferentiable = True
    dtype = numpy.float32
    grad_outputs = (True, True)
    grad_grad_inputs = (True, True)

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
            print('Hello World!')
        c = _shaped_random(self.c_shape, self.dtype)
        x = _shaped_random(self.x_shape, self.dtype)
        return (c, x)

    def forward(self, inputs, device):
        if False:
            i = 10
            return i + 15
        (c, x) = inputs
        (c, h) = F.lstm(c, x)
        return (c, h)

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        (c, x) = inputs
        batch = x.shape[0]

        def _extract_gates(x):
            if False:
                return 10
            r = x.reshape((len(x), x.shape[1] // 4, 4) + x.shape[2:])
            return [r[:, :, i] for i in six.moves.range(4)]
        (a, i, f, o) = _extract_gates(x)
        a = numpy.tanh(a)
        i = sigmoid(i)
        f = sigmoid(f)
        o = sigmoid(o)
        c_exp = numpy.zeros_like(c)
        c_exp[:batch] = a * i + f * c[:batch]
        h_exp = o * numpy.tanh(c_exp[:batch])
        c_exp[batch:] = c[batch:]
        return (c_exp, h_exp)

    def generate_grad_outputs(self, outputs_template):
        if False:
            for i in range(10):
                print('nop')
        grad_out = []
        c = outputs_template[0]
        h = outputs_template[1]
        c_shape = c.shape
        h_shape = h.shape
        if self.grad_outputs[0] is True:
            grad_out.append(_shaped_random(c_shape, c.dtype))
        else:
            grad_out.append(None)
        if self.grad_outputs[1] is True:
            grad_out.append(_shaped_random(h_shape, h.dtype))
        else:
            grad_out.append(None)
        return tuple(grad_out)

    def generate_grad_grad_inputs(self, inputs_template):
        if False:
            while True:
                i = 10
        grad_grad_in = []
        c = inputs_template[0]
        x = inputs_template[1]
        c_shape = c.shape
        x_shape = x.shape
        if self.grad_grad_inputs[0] is True:
            grad_grad_in.append(_shaped_random(c_shape, c.dtype))
        else:
            grad_grad_in.append(None)
        if self.grad_grad_inputs[1] is True:
            grad_grad_in.append(_shaped_random(x_shape, x.dtype))
        else:
            grad_grad_in.append(None)
        return tuple(grad_grad_in)

@testing.fix_random()
@backend.inject_backend_tests(None, testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0']}) + testing.product({'use_cuda': [False], 'use_ideep': ['never', 'always']}) + testing.product([[{'use_cuda': True}], testing.product({'use_cudnn': ['never']}) + testing.product({'use_cudnn': ['always'], 'cudnn_deterministic': [True, False], 'autotune': [True, False]})]))
@testing.parameterize(*testing.product_dict([{'c_shape': (10, 3), 'x_shape': (10, 12)}, {'c_shape': (20, 32, 4), 'x_shape': (16, 128, 4)}, {'c_shape': (32, 100, 3, 5), 'x_shape': (32, 400, 3, 5)}, {'c_shape': (16, 20), 'x_shape': (2, 80)}, {'c_shape': (16, 20), 'x_shape': (0, 80)}, {'c_shape': (0, 0), 'x_shape': (0, 0)}, {'c_shape': (8, 0), 'x_shape': (2, 0)}], [{'dtype': numpy.float16}, {'dtype': numpy.float32}, {'dtype': numpy.float64}]))
class TestLSTM(LSTMTestBase, testing.FunctionTestCase):
    pass

@testing.fix_random()
@backend.inject_backend_tests(None, testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0']}) + testing.product({'use_cuda': [False], 'use_ideep': ['never', 'always']}) + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always']}))
@testing.parameterize(*testing.product_dict([{'c_shape': (10, 3), 'x_shape': (10, 12)}], [{'grad_outputs': (True, True)}, {'grad_outputs': (True, False)}, {'grad_outputs': (False, True)}], [{'grad_grad_inputs': (True, True)}, {'grad_grad_inputs': (True, False)}, {'grad_grad_inputs': (False, True)}]))
class TestLSTMGradOutputs(LSTMTestBase, testing.FunctionTestCase):
    pass

@testing.parameterize(*testing.product({'batch': [3, 2, 0], 'dtype': [numpy.float32]}) + testing.product({'batch': [3], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@testing.fix_random()
@inject_backend_tests(['test_backward'])
class TestLSTMGrad(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        hidden_shape = (3, 2, 4)
        dtype = self.dtype
        x_shape = (self.batch, 8, 4)
        y_shape = (self.batch, 2, 4)
        c_prev = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        c_next = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        gc = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        gh = numpy.random.uniform(-1, 1, y_shape).astype(dtype)
        ggc_prev = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        ggx = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        self.inputs = [c_prev, x, c_next, gc, gh]
        self.grad_outputs = [ggc_prev, ggx]
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_backward_options = {'dtype': numpy.float64, 'atol': 0.001, 'rtol': 0.01}

    def check_backward(self, inputs, grad_outputs, backend_config):
        if False:
            while True:
                i = 10
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
        with backend_config:
            gradient_check.check_backward(lstm.LSTMGrad(), inputs, grad_outputs, **self.check_backward_options)

    def test_backward(self, backend_config):
        if False:
            i = 10
            return i + 15
        self.check_backward(self.inputs, self.grad_outputs, backend_config)
testing.run_module(__name__, __file__)