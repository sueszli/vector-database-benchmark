"""
Test of the activation functions
"""
from math import tanh as true_tanh
import numpy as np
from neon import NervanaObject
from neon.transforms import Identity, Rectlin, Softmax, Tanh, Logistic, Sign, PixelwiseSoftmax
from utils import allclose_with_out

def compare_tensors(func, inputs, outputs, deriv=False, tol=0.0):
    if False:
        return 10
    be = NervanaObject.be
    temp = be.empty(outputs.shape)
    dtypeu = np.float32
    if deriv is True:
        temp[:] = func.bprop(be.array(dtypeu(inputs)))
    else:
        temp[:] = func(be.array(dtypeu(inputs)))
    cond = np.sum(np.abs(temp.get() - outputs) <= tol)
    assert cond == np.prod(outputs.shape)
'Identity\n'

def test_identity(backend_default):
    if False:
        print('Hello World!')
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.array([0, 1, -2]).reshape((3, 1))
    compare_tensors(Identity(), inputs, outputs)

def test_identity_derivative(backend_default):
    if False:
        while True:
            i = 10
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.ones((1, 1))
    compare_tensors(Identity(), inputs, outputs, deriv=True)
'Rectified Linear unit\n'

def test_rectlin_positives(backend_default):
    if False:
        i = 10
        return i + 15
    inputs = np.array([1, 3, 2]).reshape((3, 1))
    outputs = np.array([1, 3, 2]).reshape((3, 1))
    compare_tensors(Rectlin(), inputs, outputs)

def test_rectlin_negatives(backend_default):
    if False:
        i = 10
        return i + 15
    inputs = np.array([[-1, -3], [-2, -4]])
    outputs = np.array([[0, 0], [0, 0]])
    compare_tensors(Rectlin(), inputs, outputs)

def test_rectlin_mixed(backend_default):
    if False:
        while True:
            i = 10
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[4, 0], [0, 9]])
    compare_tensors(Rectlin(), inputs, outputs)

def test_rectlin_derivative_positives(backend_default):
    if False:
        i = 10
        return i + 15
    inputs = np.array([1, 3, 2]).reshape((3, 1))
    outputs = np.array([1, 1, 1]).reshape((3, 1))
    compare_tensors(Rectlin(), inputs, outputs, deriv=True)

def test_rectlin_derivative_negatives(backend_default):
    if False:
        while True:
            i = 10
    inputs = np.array([[-1, -3], [-2, -4]])
    outputs = np.array([[0, 0], [0, 0]])
    compare_tensors(Rectlin(), inputs, outputs, deriv=True)

def test_rectlin_derivative_mixed(backend_default):
    if False:
        i = 10
        return i + 15
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[1, 0], [0, 1]])
    compare_tensors(Rectlin(), inputs, outputs, deriv=True)
'Leaky Rectified Linear unit\n'

def test_leaky_rectlin_positives(backend_default):
    if False:
        while True:
            i = 10
    slope = 0.2
    inputs = np.array([1, 3, 2]).reshape((3, 1))
    outputs = np.array([1, 3, 2]).reshape((3, 1))
    compare_tensors(Rectlin(slope=slope), inputs, outputs)

def test_leaky_rectlin_negatives(backend_default):
    if False:
        for i in range(10):
            print('nop')
    slope = 0.2
    inputs = np.array([[-1, -3], [-2, -4]])
    outputs = inputs * slope
    compare_tensors(Rectlin(slope=slope), inputs, outputs, tol=1e-07)

def test_leaky_rectlin_mixed(backend_default):
    if False:
        for i in range(10):
            print('nop')
    slope = 0.2
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[4, 0], [-2 * slope, 9]])
    compare_tensors(Rectlin(slope=slope), inputs, outputs, tol=1e-07)

def test_leaky_rectlin_derivative_positives(backend_default):
    if False:
        for i in range(10):
            print('nop')
    slope = 0.2
    inputs = np.array([1, 3, 2]).reshape((3, 1))
    outputs = np.array([1, 1, 1]).reshape((3, 1))
    compare_tensors(Rectlin(slope=slope), inputs, outputs, deriv=True)

def test_leaky_rectlin_derivative_negatives(backend_default):
    if False:
        return 10
    slope = 0.2
    inputs = np.array([[-1, -3], [-2, -4]])
    outputs = np.array([[0, 0], [0, 0]]) + slope
    compare_tensors(Rectlin(slope=slope), inputs, outputs, deriv=True, tol=1e-07)

def test_leaky_rectlin_derivative_mixed(backend_default):
    if False:
        for i in range(10):
            print('nop')
    slope = 0.2
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[1, 0], [slope, 1]])
    compare_tensors(Rectlin(slope=slope), inputs, outputs, deriv=True, tol=1e-07)
'Softmax\n'

def test_softmax(backend_default):
    if False:
        return 10
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.exp(inputs - 1) / np.sum(np.exp(inputs - 1))
    compare_tensors(Softmax(), inputs, outputs, tol=1e-07)

def test_softmax_derivative(backend_default):
    if False:
        return 10
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.ones((1, 1))
    compare_tensors(Softmax(), inputs, outputs, deriv=True)

def test_softmax_big_inputs(backend_default):
    if False:
        i = 10
        return i + 15
    np.random.seed(1)
    be = backend_default
    assert be.bsz >= 128, 'This tests needs large batch size'
    act = Softmax()
    Nout = 1000
    x_ = np.random.random((Nout, be.bsz))
    x = be.iobuf(Nout)
    x[:] = x_
    mx = np.max(x_, axis=0)
    ex = np.exp(x_ - mx)
    y_ = ex / np.sum(ex, axis=0)
    x[:] = act(x)
    assert allclose_with_out(y_, x.get(), atol=0.0, rtol=1e-05)
'PixelwiseSoftmax\n'

def test_pixelwise_softmax(backend_default):
    if False:
        for i in range(10):
            print('nop')
    inputs = np.array([0, 1, 3, 1, 2, -2]).reshape((2, 3))
    outputs = 1.0 / np.sum(np.exp(inputs - np.max(inputs, axis=0)), axis=0) * np.exp(inputs - np.max(inputs, axis=0))
    inputs = inputs.reshape((1, -1))
    outputs = outputs.reshape((1, -1))
    compare_tensors(PixelwiseSoftmax(c=2), inputs, outputs, deriv=False, tol=1e-06)

def test_pixelwise_softmax_derivative(backend_default):
    if False:
        while True:
            i = 10
    inputs = np.array([0, 1, 3, 1, 2, -2]).reshape((2, 3))
    outputs = np.ones((1, 6))
    inputs = inputs.reshape((1, -1))
    outputs = outputs.reshape((1, -1))
    compare_tensors(PixelwiseSoftmax(c=2), inputs, outputs, deriv=True, tol=1e-06)
'Tanh\n'

def test_tanh(backend_default):
    if False:
        return 10
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.array([true_tanh(0), true_tanh(1), true_tanh(-2)]).reshape((3, 1))
    compare_tensors(Tanh(), inputs, outputs, tol=1e-07)

def test_tanh_derivative(backend_default):
    if False:
        print('Hello World!')
    inputs = np.array([true_tanh(0), true_tanh(1), true_tanh(-2)]).reshape((3, 1))
    outputs = np.array([1 - true_tanh(0) ** 2, 1 - true_tanh(1) ** 2, 1 - true_tanh(-2) ** 2]).reshape((3, 1))
    compare_tensors(Tanh(), inputs, outputs, deriv=True, tol=1e-07)
'Logistic\n'

def test_logistic(backend_default):
    if False:
        while True:
            i = 10
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    compare_tensors(Logistic(), inputs, outputs, tol=1e-07)

def test_logistic_derivative(backend_default):
    if False:
        print('Hello World!')
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    inputs = 1.0 / (1.0 + np.exp(-inputs))
    outputs = inputs * (1.0 - inputs)
    compare_tensors(Logistic(shortcut=False), inputs, outputs, deriv=True, tol=1e-07)
'Sign\n'

def test_sign(backend_default):
    if False:
        for i in range(10):
            print('nop')
    inputs = np.array([-1, -0.5, 0, 0.5, 1]).reshape((5, 1))
    outputs = np.array([-1, -1, 1, 1, 1]).reshape((5, 1))
    compare_tensors(Sign(), inputs, outputs, tol=0)