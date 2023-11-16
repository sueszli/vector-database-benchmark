import inspect
import sys
import unittest
import numpy
import pytest
import chainer
import chainer.testing
import chainerx

class OpTest(chainer.testing.function_link.FunctionTestBase):
    """Base class for op test.

    It must be used in conjunction with `op_test` decorator.

    Examples:

    @op_utils.op_test(['native:0', 'cuda:0'])
    class test_relu(op_utils.OpTest):

        # ReLU function has a non-differentiable point around zero, so
        # dodge_nondifferentiable should be set to True.
        dodge_nondifferentiable = True

        def setup(self, float_dtype):
            self.dtype = float_dtype

        def generate_inputs(self):
            dtype = self.dtype
            x = numpy.random.uniform(-1, 1, (1, 3)).astype(dtype)
            return x, w, b

        def forward_chainerx(self, inputs):
            x, w, b = inputs
            y = chainerx.relu(x)
            return y,

        def forward_expected(self, inputs):
            x, w, b = inputs
            expected = x.copy()
            expected[expected < 0] = 0
            return expected,

    In this example, `float_dtype` is a Pytest fixture for parameterizing
    floating-point dtypes (i.e. float16, float32, float64). As seen from
    this, arguments in the `setup` method are treated as Pytest fixtures.

    Test implementations must at least override the following methods:
      * `generate_inputs`: Generates inputs to the test target.
      * `forward_chainerx`: Forward implementation using ChainerX.
      * `forward_expected`: Forward reference implementation.

    It can have the same attributes as `chainer.testing.FunctionTestCase`.
    """

    def setup(self):
        if False:
            while True:
                i = 10
        pass

    def teardown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def forward(self, inputs, device):
        if False:
            i = 10
            return i + 15
        test_self = self

        class MyFunc(chainer.FunctionNode):

            def forward_chainerx(self, inputs):
                if False:
                    i = 10
                    return i + 15
                return test_self.forward_chainerx(inputs)
        return MyFunc().apply(inputs)

    def forward_chainerx(self, inputs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Op test implementation must override `forward_chainerx`.')

    def run_test_forward(self, backend_config):
        if False:
            while True:
                i = 10
        if self.skip_forward_test:
            raise unittest.SkipTest('skip_forward_test is set')
        super(OpTest, self).run_test_forward(backend_config)

    def run_test_backward(self, backend_config):
        if False:
            print('Hello World!')
        if self.skip_backward_test:
            return
        super(OpTest, self).run_test_backward(backend_config)

    def run_test_double_backward(self, backend_config):
        if False:
            for i in range(10):
                print('nop')
        if self.skip_double_backward_test:
            return
        super(OpTest, self).run_test_double_backward(backend_config)

class ChainerOpTest(OpTest):
    """Base class for op test that compares the output with Chainer
    implementation.

    It must be used in conjunction with `op_test` decorator.

    Examples:

    @op_utils.op_test(['native:0', 'cuda:0'])
    class test_conv(op_utils.ChainerOpTest):

        def setup(self, float_dtype):
            self.dtype = float_dtype

        def generate_inputs(self):
            dtype = self.dtype
            x = numpy.random.uniform(-1, 1, (1, 3)).astype(dtype)
            w = numpy.random.uniform(-1, 1, (5, 3)).astype(dtype)
            b = numpy.random.uniform(-1, 1, (5,)).astype(dtype)
            return x, w, b

        def forward_chainerx(self, inputs):
            x, w, b = inputs
            y = chainerx.conv(x, w, b, self.stride, self.pad, self.cover_all)
            return y,

        def forward_chainer(self, inputs):
            x, w, b = inputs
            y = chainer.functions.convolution_nd(
                x, w, b, self.stride, self.pad, self.cover_all)
            return y,

    In this example, `float_dtype` is a Pytest fixture for parameterizing
    floating-point dtypes (i.e. float16, float32, float64). As seen from
    this, arguments in the `setup` method are treated as Pytest fixtures.

    Test implementations must at least override the following methods:
      * `generate_inputs`: Generates inputs to the test target.
      * `forward_chainerx`: Forward implementation using ChainerX.
      * `forward_chainer`: Forward reference implementation using Chainer.

    It can have the same attributes as `chainer.testing.FunctionTestCase`.
    """

    def forward_expected(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        output_vars = self.forward_chainer(inputs)
        return tuple([y.array for y in output_vars])

    def forward_chainerx(self, inputs):
        if False:
            print('Hello World!')
        raise NotImplementedError('Op test implementation must override `forward_chainerx`.')

    def forward_chainer(self, inputs):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Op test implementation must override `forward_chainer`.')

class NumpyOpTest(OpTest):
    """Base class for op test that compares the output with NumPy
    implementation.

    It must be used in conjunction with `op_test` decorator.

    Examples:

    @op_utils.op_test(['native:0', 'cuda:0'])
    class test_tanh(op_utils.NumpyOpTest):

        def setup(self, float_dtype):
            self.dtype = dtype

        def generate_inputs(self):
            x = numpy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
            return x,

        def forward_xp(self, inputs, xp):
            x, = inputs
            return xp.tanh(x),

    In this example, `float_dtype` is a Pytest fixture for parameterizing
    floating-point dtypes (i.e. float16, float32, float64). As seen from
    this, arguments in the `setup` method are treated as Pytest fixtures.

    Test implementations must at least override the following methods:
      * `generate_inputs`: Generates inputs to the test target.
      * `forward_xp`: Forward implementation using both ChainerX and NumPy.

    It can have the same attributes as `chainer.testing.FunctionTestCase`.

    This test also compares strides of forward output arrays with NumPy
    outputs. Set ``check_numpy_strides_compliance`` attribute to ``False``
    to skip this check.

    Acceptable errors in forward computations can be configured with
    ``forward_accept_errors``. If both ChainerX/NumPy forward implementations
    raise one of those errors, the test will succeed and
    backward/double-backward tests will be skipped.
    """
    check_numpy_strides_compliance = True
    forward_accept_errors = ()
    __forward_error_chainerx = None
    __forward_error_expected = None

    @property
    def is_forward_successful_with_accept_errors(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__forward_error_chainerx not in (None, 'ok') and self.__forward_error_expected not in (None, 'ok')

    def __get_accept_errors(self):
        if False:
            print('Hello World!')
        if self.test_name == 'test_forward':
            return self.forward_accept_errors
        return ()

    def forward_chainerx(self, inputs):
        if False:
            return 10
        accept_errors = self.__get_accept_errors()
        try:
            outputs = self.forward_xp(inputs, chainerx)
            self.__forward_error_chainerx = 'ok'
        except accept_errors as e:
            self.__forward_error_chainerx = e
            y = chainerx.zeros((0,), 'float32')
            outputs = (y,)
        return outputs

    def forward_expected(self, inputs):
        if False:
            while True:
                i = 10
        accept_errors = self.__get_accept_errors()
        try:
            outputs = self.forward_xp(inputs, numpy)
            self.__forward_error_expected = 'ok'
        except accept_errors as e:
            self.__forward_error_expected = e
            y = numpy.zeros((0,), 'float32')
            outputs = (y,)
        return tuple([numpy.asarray(y) for y in outputs])

    def forward_xp(self, inputs, xp):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Op test implementation must override `forward_xp`.')

    def check_forward_outputs(self, outputs, expected_outputs):
        if False:
            return 10
        error_chainerx = self.__forward_error_chainerx
        error_expected = self.__forward_error_expected
        assert error_chainerx is not None
        assert error_expected is not None
        if not (error_chainerx == 'ok' and error_expected == 'ok'):
            if error_chainerx == 'ok':
                chainer.testing.FunctionTestError.fail('Error raised in NumPy while not in ChainerX.', error_expected)
            if error_expected == 'ok':
                chainer.testing.FunctionTestError.fail('Error raised in ChainerX while not in NumPy.', error_chainerx)
            return
        assert error_chainerx == 'ok'
        assert error_expected == 'ok'
        super(NumpyOpTest, self).check_forward_outputs(outputs, expected_outputs)
        if self.check_numpy_strides_compliance:
            if not all((a.strides == e.strides for (a, e) in zip(outputs, expected_outputs))):
                msg = 'Strides do not match with NumPy outputs.\nExpected shapes and dtypes: {}\nActual shapes and dtypes:   {}\nExpected strides: {}\nActual strides:   {}\n'.format(chainer.utils._format_array_props(expected_outputs), chainer.utils._format_array_props(outputs), ', '.join((str(e.strides) for e in expected_outputs)), ', '.join((str(a.strides) for a in outputs)))
                chainer.testing.FunctionTestError.fail(msg)

def _make_backend_config(device_name):
    if False:
        i = 10
        return i + 15
    backend_config = chainer.testing.BackendConfig({'use_chainerx': True, 'chainerx_device': device_name})
    return backend_config

def _create_test_entry_function(cls, module, devices):
    if False:
        print('Hello World!')
    if not cls.__name__.startswith('Test'):
        raise TypeError("OpTest class name must start with 'Test'. Actual: {!r}".format(cls.__name__))
    func_name = 'test_{}'.format(cls.__name__[len('Test'):])

    @pytest.mark.parametrize_device(devices)
    def entry_func(device, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        backend_config = _make_backend_config(device.name)
        obj = cls()
        try:
            obj.setup(*args, **kwargs)
            obj.run_test_forward(backend_config)
        finally:
            obj.teardown()
        if isinstance(obj, NumpyOpTest):
            if obj.is_forward_successful_with_accept_errors:
                return
        obj = cls()
        try:
            obj.setup(*args, **kwargs)
            obj.run_test_backward(backend_config)
        finally:
            obj.teardown()
        obj = cls()
        try:
            obj.setup(*args, **kwargs)
            obj.run_test_double_backward(backend_config)
        finally:
            obj.teardown()
    entry_func.__name__ = func_name
    sig = inspect.signature(cls.setup)
    params = list(sig.parameters.values())
    params = params[1:]
    device_param = inspect.Parameter('device', inspect.Parameter.POSITIONAL_OR_KEYWORD)
    params = [device_param] + params
    entry_func.__signature__ = inspect.Signature(params)
    try:
        pytestmark = cls.pytestmark
        entry_func.pytestmark += pytestmark
    except AttributeError:
        pass
    setattr(module, func_name, entry_func)

def op_test(devices):
    if False:
        while True:
            i = 10
    'Decorator to set up an op test.\n\n    This decorator can be used in conjunction with either ``NumpyOpTest`` or\n    ``ChainerOpTest`` to define an op test.\n\n    See the documentation of the respective classes for detailed explanation\n    and examples.\n\n    Args:\n        devices: List of devices to test.\n    '

    def wrap(cls):
        if False:
            i = 10
            return i + 15
        if isinstance(cls, chainer.testing._bundle._ParameterizedTestCaseBundle):
            classes = []
            for (c, m, name) in cls.cases:
                classes.append((c, m))
                if m is not None:
                    delattr(sys.modules[m], name)
        else:
            classes = [(cls, cls.__module__)]
        for (cls, mod) in classes:
            if not issubclass(cls, OpTest):
                raise TypeError('@op_test decorator can only be applied to OpTest class definition.')
            _create_test_entry_function(cls, sys.modules[mod], devices)
        return None
    return wrap

def fix_random():
    if False:
        while True:
            i = 10
    'Decorator that fixes random numbers in an op test.\n\n    .. seealso:: :func:`~chainer.testing.fix_random`\n    '
    return chainer.testing.random.fix_random(setup_method='setup', teardown_method='teardown')