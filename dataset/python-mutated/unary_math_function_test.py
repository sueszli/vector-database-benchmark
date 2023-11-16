import unittest
import warnings
import numpy
from chainer.backends import cuda
from chainer import function
from chainer import functions
from chainer import variable
try:
    from chainer.testing import attr
    _error = attr.get_error()
except ImportError as e:
    _error = e

def is_available():
    if False:
        print('Hello World!')
    return _error is None

def check_available():
    if False:
        while True:
            i = 10
    if _error is not None:
        raise RuntimeError('{} is not available.\n\nReason: {}: {}'.format(__name__, type(_error).__name__, _error))

def _func_name(func):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(func, function.Function):
        return func.__class__.__name__.lower()
    else:
        return func.__name__

def _func_class(func):
    if False:
        print('Hello World!')
    if isinstance(func, function.Function):
        return func.__class__
    else:
        name = func.__name__.capitalize()
        return getattr(functions, name, None)

def _make_data_default(shape, dtype):
    if False:
        while True:
            i = 10
    x = numpy.random.uniform(-1, 1, shape).astype(dtype, copy=False)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype, copy=False)
    ggx = numpy.random.uniform(-1, 1, shape).astype(dtype, copy=False)
    return (x, gy, ggx)

def _nonlinear(func):
    if False:
        return 10

    def aux(x):
        if False:
            for i in range(10):
                print('nop')
        y = func(x)
        return y * y
    return aux

def unary_math_function_unittest(func, func_expected=None, label_expected=None, make_data=None, is_linear=None, forward_options=None, backward_options=None, double_backward_options=None):
    if False:
        i = 10
        return i + 15
    "Decorator for testing unary mathematical Chainer functions.\n\n    This decorator makes test classes test unary mathematical Chainer\n    functions. Tested are forward and backward, including double backward,\n    computations on CPU and GPU across parameterized ``shape`` and ``dtype``.\n\n    Args:\n        func(function or ~chainer.Function): Chainer function to be tested by\n            the decorated test class. Taking :class:`~chainer.Function` is for\n            backward compatibility.\n        func_expected: Function used to provide expected values for\n            testing forward computation. If not given, a corresponsing numpy\n            function for ``func`` is implicitly picked up by its name.\n        label_expected(string): String used to test labels of Chainer\n            functions. If not given, the name of ``func`` is implicitly used.\n        make_data: Function to customize input and gradient data used\n            in the tests. It takes ``shape`` and ``dtype`` as its arguments,\n            and returns a tuple of input, gradient and double gradient data. By\n            default, uniform destribution ranged ``[-1, 1]`` is used for all of\n            them.\n        is_linear: Tells the decorator that ``func`` is a linear function\n            so that it wraps ``func`` as a non-linear function to perform\n            double backward test. This argument is left for backward\n            compatibility. Linear functions can be tested by default without\n            specifying ``is_linear`` in Chainer v5 or later.\n        forward_options(dict): Options to be specified as an argument of\n            :func:`chainer.testing.assert_allclose` function.\n            If not given, preset tolerance values are automatically selected.\n        backward_options(dict): Options to be specified as an argument of\n            :func:`chainer.gradient_check.check_backward` function.\n            If not given, preset tolerance values are automatically selected\n            depending on ``dtype``.\n        double_backward_options(dict): Options to be specified as an argument\n            of :func:`chainer.gradient_check.check_double_backward` function.\n            If not given, preset tolerance values are automatically selected\n            depending on ``dtype``.\n\n    The decorated test class tests forward, backward and double backward\n    computations on CPU and GPU across the following\n    :func:`~chainer.testing.parameterize` ed parameters:\n\n    - shape: rank of zero, and rank of more than zero\n    - dtype: ``numpy.float16``, ``numpy.float32`` and ``numpy.float64``\n\n    Additionally, it tests the label of the Chainer function.\n\n    Chainer functions tested by the test class decorated with the decorator\n    should have the following properties:\n\n    - Unary, taking one parameter and returning one value\n    - ``dtype`` of input and output are the same\n    - Elementwise operation for the supplied ndarray\n\n    .. admonition:: Example\n\n       The following code defines a test class that tests\n       :func:`~chainer.functions.sin` Chainer function, which takes a parameter\n       with ``dtype`` of float and returns a value with the same ``dtype``.\n\n       .. doctest::\n\n          >>> import unittest\n          >>> from chainer import testing\n          >>> from chainer import functions as F\n          >>>\n          >>> @testing.unary_math_function_unittest(F.sin)\n          ... class TestSin(unittest.TestCase):\n          ...     pass\n\n       Because the test methods are implicitly injected to ``TestSin`` class by\n       the decorator, it is enough to place ``pass`` in the class definition.\n\n       To customize test data, ``make_data`` optional parameter can be used.\n       The following is an example of testing ``sqrt`` Chainer function, which\n       is tested in positive value domain here instead of the default input.\n\n       .. doctest::\n\n          >>> import numpy\n          >>>\n          >>> def make_data(shape, dtype):\n          ...     x = numpy.random.uniform(0.1, 1, shape).astype(dtype)\n          ...     gy = numpy.random.uniform(-1, 1, shape).astype(dtype)\n          ...     ggx = numpy.random.uniform(-1, 1, shape).astype(dtype)\n          ...     return x, gy, ggx\n          ...\n          >>> @testing.unary_math_function_unittest(F.sqrt,\n          ...                                       make_data=make_data)\n          ... class TestSqrt(unittest.TestCase):\n          ...     pass\n\n       ``make_data`` function which returns input, gradient and double gradient\n       data generated in proper value domains with given ``shape`` and\n       ``dtype`` parameters is defined, then passed to the decorator's\n       ``make_data`` parameter.\n\n    "
    check_available()
    from chainer import gradient_check
    from chainer import testing
    is_new_style = not isinstance(func, function.Function)
    func_name = _func_name(func)
    func_class = _func_class(func)
    if func_expected is None:
        try:
            func_expected = getattr(numpy, func_name)
        except AttributeError:
            raise ValueError("NumPy has no functions corresponding to Chainer function '{}'.".format(func_name))
    if label_expected is None:
        label_expected = func_name
    elif func_class is None:
        raise ValueError('Expected label is given even though Chainer function does not have its label.')
    if make_data is None:
        if is_new_style:
            make_data = _make_data_default
        else:

            def aux(shape, dtype):
                if False:
                    i = 10
                    return i + 15
                return _make_data_default(shape, dtype)[0:2]
            make_data = aux
    if is_linear is not None:
        warnings.warn('is_linear option is deprecated', DeprecationWarning)

    def f(klass):
        if False:
            for i in range(10):
                print('nop')
        assert issubclass(klass, unittest.TestCase)

        def setUp(self):
            if False:
                while True:
                    i = 10
            if is_new_style:
                (self.x, self.gy, self.ggx) = make_data(self.shape, self.dtype)
            else:
                (self.x, self.gy) = make_data(self.shape, self.dtype)
            if self.dtype == numpy.float16:
                self.forward_options = {'atol': numpy.finfo('float16').eps, 'rtol': numpy.finfo('float16').eps}
                self.backward_options = {'eps': 2 ** (-4), 'atol': 2 ** (-4), 'rtol': 2 ** (-4), 'dtype': numpy.float64}
                self.double_backward_options = {'eps': 2 ** (-4), 'atol': 2 ** (-4), 'rtol': 2 ** (-4), 'dtype': numpy.float64}
            else:
                self.forward_options = {'atol': 0.0001, 'rtol': 0.0001}
                self.backward_options = {'dtype': numpy.float64, 'atol': 0.0001, 'rtol': 0.0001}
                self.double_backward_options = {'dtype': numpy.float64, 'atol': 0.0001, 'rtol': 0.0001}
            if forward_options is not None:
                self.forward_options.update(forward_options)
            if backward_options is not None:
                self.backward_options.update(backward_options)
            if double_backward_options is not None:
                self.double_backward_options.update(double_backward_options)
        setattr(klass, 'setUp', setUp)

        def check_forward(self, x_data):
            if False:
                return 10
            x = variable.Variable(x_data)
            y = func(x)
            self.assertEqual(y.data.dtype, x_data.dtype)
            y_expected = func_expected(cuda.to_cpu(x_data), dtype=x_data.dtype)
            testing.assert_allclose(y_expected, y.data, **self.forward_options)
        setattr(klass, 'check_forward', check_forward)

        def test_forward_cpu(self):
            if False:
                i = 10
                return i + 15
            self.check_forward(self.x)
        setattr(klass, 'test_forward_cpu', test_forward_cpu)

        @attr.gpu
        def test_forward_gpu(self):
            if False:
                return 10
            self.check_forward(cuda.to_gpu(self.x))
        setattr(klass, 'test_forward_gpu', test_forward_gpu)

        def check_backward(self, x_data, y_grad):
            if False:
                return 10
            gradient_check.check_backward(func, x_data, y_grad, **self.backward_options)
        setattr(klass, 'check_backward', check_backward)

        def test_backward_cpu(self):
            if False:
                print('Hello World!')
            self.check_backward(self.x, self.gy)
        setattr(klass, 'test_backward_cpu', test_backward_cpu)

        @attr.gpu
        def test_backward_gpu(self):
            if False:
                i = 10
                return i + 15
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))
        setattr(klass, 'test_backward_gpu', test_backward_gpu)
        if is_new_style:

            def check_double_backward(self, x_data, y_grad, x_grad_grad):
                if False:
                    print('Hello World!')
                func1 = _nonlinear(func) if is_linear else func
                gradient_check.check_double_backward(func1, x_data, y_grad, x_grad_grad, **self.double_backward_options)
            setattr(klass, 'check_double_backward', check_double_backward)

            def test_double_backward_cpu(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.check_double_backward(self.x, self.gy, self.ggx)
            setattr(klass, 'test_double_backward_cpu', test_double_backward_cpu)

            @attr.gpu
            def test_double_backward_gpu(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))
            setattr(klass, 'test_double_backward_gpu', test_double_backward_gpu)
        if func_class is not None:

            def test_label(self):
                if False:
                    while True:
                        i = 10
                self.assertEqual(func_class().label, label_expected)
            setattr(klass, 'test_label', test_label)
        return testing.parameterize(*testing.product({'shape': [(3, 2), ()], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))(klass)
    return f