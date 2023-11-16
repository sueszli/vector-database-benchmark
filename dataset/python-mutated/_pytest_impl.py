import unittest
import cupy.testing._parameterized
try:
    import pytest
    import _pytest
    _error = None
except ImportError as e:
    pytest = None
    _pytest = None
    _error = e

def is_available():
    if False:
        return 10
    return _error is None and hasattr(pytest, 'fixture')

def check_available(feature):
    if False:
        i = 10
        return i + 15
    if not is_available():
        raise RuntimeError('cupy.testing: {} is not available.\n\nReason: {}: {}'.format(feature, type(_error).__name__, _error))
if is_available():

    class _TestingParameterizeMixin:

        def __repr__(self):
            if False:
                i = 10
                return i + 15
            return '<{}  parameter: {}>'.format(super().__repr__(), self.__dict__)

        @pytest.fixture(autouse=True)
        def _cupy_testing_parameterize(self, _cupy_testing_param):
            if False:
                print('Hello World!')
            assert not self.__dict__, 'There should not be another hack with instance attribute.'
            self.__dict__.update(_cupy_testing_param)

def parameterize(*params, _ids=True):
    if False:
        while True:
            i = 10
    check_available('parameterize')
    if _ids:
        param_name = cupy.testing._parameterized._make_class_name
    else:

        def param_name(_, i, param):
            if False:
                print('Hello World!')
            return str(i)
    params = [pytest.param(param, id=param_name('', i, param)) for (i, param) in enumerate(params)]

    def f(cls):
        if False:
            i = 10
            return i + 15
        assert not issubclass(cls, unittest.TestCase)
        if issubclass(cls, _TestingParameterizeMixin):
            raise RuntimeError('do not `@testing.parameterize` twice')
        module_name = cls.__module__
        cls = type(cls.__name__, (_TestingParameterizeMixin, cls), {})
        cls.__module__ = module_name
        cls = pytest.mark.parametrize('_cupy_testing_param', params)(cls)
        return cls
    return f