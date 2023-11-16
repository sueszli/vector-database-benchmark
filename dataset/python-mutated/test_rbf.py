""" Test functions for rbf module """
import numpy as np
from numpy.testing import assert_, assert_array_almost_equal, assert_almost_equal
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
FUNCTIONS = ('multiquadric', 'inverse multiquadric', 'gaussian', 'cubic', 'quintic', 'thin-plate', 'linear')

def check_rbf1d_interpolation(function):
    if False:
        while True:
            i = 10
    x = linspace(0, 10, 9)
    y = sin(x)
    rbf = Rbf(x, y, function=function)
    yi = rbf(x)
    assert_array_almost_equal(y, yi)
    assert_almost_equal(rbf(float(x[0])), y[0])

def check_rbf2d_interpolation(function):
    if False:
        for i in range(10):
            print('nop')
    x = random.rand(50, 1) * 4 - 2
    y = random.rand(50, 1) * 4 - 2
    z = x * exp(-x ** 2 - 1j * y ** 2)
    rbf = Rbf(x, y, z, epsilon=2, function=function)
    zi = rbf(x, y)
    zi.shape = x.shape
    assert_array_almost_equal(z, zi)

def check_rbf3d_interpolation(function):
    if False:
        i = 10
        return i + 15
    x = random.rand(50, 1) * 4 - 2
    y = random.rand(50, 1) * 4 - 2
    z = random.rand(50, 1) * 4 - 2
    d = x * exp(-x ** 2 - y ** 2)
    rbf = Rbf(x, y, z, d, epsilon=2, function=function)
    di = rbf(x, y, z)
    di.shape = x.shape
    assert_array_almost_equal(di, d)

def test_rbf_interpolation():
    if False:
        i = 10
        return i + 15
    for function in FUNCTIONS:
        check_rbf1d_interpolation(function)
        check_rbf2d_interpolation(function)
        check_rbf3d_interpolation(function)

def check_2drbf1d_interpolation(function):
    if False:
        for i in range(10):
            print('nop')
    x = linspace(0, 10, 9)
    y0 = sin(x)
    y1 = cos(x)
    y = np.vstack([y0, y1]).T
    rbf = Rbf(x, y, function=function, mode='N-D')
    yi = rbf(x)
    assert_array_almost_equal(y, yi)
    assert_almost_equal(rbf(float(x[0])), y[0])

def check_2drbf2d_interpolation(function):
    if False:
        while True:
            i = 10
    x = random.rand(50) * 4 - 2
    y = random.rand(50) * 4 - 2
    z0 = x * exp(-x ** 2 - 1j * y ** 2)
    z1 = y * exp(-y ** 2 - 1j * x ** 2)
    z = np.vstack([z0, z1]).T
    rbf = Rbf(x, y, z, epsilon=2, function=function, mode='N-D')
    zi = rbf(x, y)
    zi.shape = z.shape
    assert_array_almost_equal(z, zi)

def check_2drbf3d_interpolation(function):
    if False:
        i = 10
        return i + 15
    x = random.rand(50) * 4 - 2
    y = random.rand(50) * 4 - 2
    z = random.rand(50) * 4 - 2
    d0 = x * exp(-x ** 2 - y ** 2)
    d1 = y * exp(-y ** 2 - x ** 2)
    d = np.vstack([d0, d1]).T
    rbf = Rbf(x, y, z, d, epsilon=2, function=function, mode='N-D')
    di = rbf(x, y, z)
    di.shape = d.shape
    assert_array_almost_equal(di, d)

def test_2drbf_interpolation():
    if False:
        return 10
    for function in FUNCTIONS:
        check_2drbf1d_interpolation(function)
        check_2drbf2d_interpolation(function)
        check_2drbf3d_interpolation(function)

def check_rbf1d_regularity(function, atol):
    if False:
        while True:
            i = 10
    x = linspace(0, 10, 9)
    y = sin(x)
    rbf = Rbf(x, y, function=function)
    xi = linspace(0, 10, 100)
    yi = rbf(xi)
    msg = 'abs-diff: %f' % abs(yi - sin(xi)).max()
    assert_(allclose(yi, sin(xi), atol=atol), msg)

def test_rbf_regularity():
    if False:
        while True:
            i = 10
    tolerances = {'multiquadric': 0.1, 'inverse multiquadric': 0.15, 'gaussian': 0.15, 'cubic': 0.15, 'quintic': 0.1, 'thin-plate': 0.1, 'linear': 0.2}
    for function in FUNCTIONS:
        check_rbf1d_regularity(function, tolerances.get(function, 0.01))

def check_2drbf1d_regularity(function, atol):
    if False:
        for i in range(10):
            print('nop')
    x = linspace(0, 10, 9)
    y0 = sin(x)
    y1 = cos(x)
    y = np.vstack([y0, y1]).T
    rbf = Rbf(x, y, function=function, mode='N-D')
    xi = linspace(0, 10, 100)
    yi = rbf(xi)
    msg = 'abs-diff: %f' % abs(yi - np.vstack([sin(xi), cos(xi)]).T).max()
    assert_(allclose(yi, np.vstack([sin(xi), cos(xi)]).T, atol=atol), msg)

def test_2drbf_regularity():
    if False:
        while True:
            i = 10
    tolerances = {'multiquadric': 0.1, 'inverse multiquadric': 0.15, 'gaussian': 0.15, 'cubic': 0.15, 'quintic': 0.1, 'thin-plate': 0.15, 'linear': 0.2}
    for function in FUNCTIONS:
        check_2drbf1d_regularity(function, tolerances.get(function, 0.01))

def check_rbf1d_stability(function):
    if False:
        return 10
    np.random.seed(1234)
    x = np.linspace(0, 10, 50)
    z = x + 4.0 * np.random.randn(len(x))
    rbf = Rbf(x, z, function=function)
    xi = np.linspace(0, 10, 1000)
    yi = rbf(xi)
    assert_(np.abs(yi - xi).max() / np.abs(z - x).max() < 1.1)

def test_rbf_stability():
    if False:
        i = 10
        return i + 15
    for function in FUNCTIONS:
        check_rbf1d_stability(function)

def test_default_construction():
    if False:
        return 10
    x = linspace(0, 10, 9)
    y = sin(x)
    rbf = Rbf(x, y)
    yi = rbf(x)
    assert_array_almost_equal(y, yi)

def test_function_is_callable():
    if False:
        print('Hello World!')
    x = linspace(0, 10, 9)
    y = sin(x)

    def linfunc(x):
        if False:
            print('Hello World!')
        return x
    rbf = Rbf(x, y, function=linfunc)
    yi = rbf(x)
    assert_array_almost_equal(y, yi)

def test_two_arg_function_is_callable():
    if False:
        i = 10
        return i + 15

    def _func(self, r):
        if False:
            for i in range(10):
                print('nop')
        return self.epsilon + r
    x = linspace(0, 10, 9)
    y = sin(x)
    rbf = Rbf(x, y, function=_func)
    yi = rbf(x)
    assert_array_almost_equal(y, yi)

def test_rbf_epsilon_none():
    if False:
        i = 10
        return i + 15
    x = linspace(0, 10, 9)
    y = sin(x)
    Rbf(x, y, epsilon=None)

def test_rbf_epsilon_none_collinear():
    if False:
        i = 10
        return i + 15
    x = [1, 2, 3]
    y = [4, 4, 4]
    z = [5, 6, 7]
    rbf = Rbf(x, y, z, epsilon=None)
    assert_(rbf.epsilon > 0)