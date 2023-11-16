import datetime
import mock
import random
import sacred.optional as opt
from sacred.config.captured_function import create_captured_function
from sacred.settings import SETTINGS

def test_create_captured_function():
    if False:
        print('Hello World!')

    def foo():
        if False:
            for i in range(10):
                print('nop')
        'my docstring'
        return 42
    cf = create_captured_function(foo)
    assert cf.__name__ == 'foo'
    assert cf.__doc__ == 'my docstring'
    assert cf.prefix is None
    assert cf.config == {}
    assert not cf.uses_randomness
    assert callable(cf)

def test_call_captured_function():
    if False:
        i = 10
        return i + 15

    def foo(a, b, c, d=4, e=5, f=6):
        if False:
            return 10
        return (a, b, c, d, e, f)
    cf = create_captured_function(foo)
    cf.logger = mock.MagicMock()
    cf.config = {'a': 11, 'b': 12, 'd': 14}
    assert cf(21, c=23, f=26) == (21, 12, 23, 14, 5, 26)
    cf.logger.debug.assert_has_calls([mock.call('Started'), mock.call('Finished after %s.', datetime.timedelta(0))])

def test_captured_function_randomness():
    if False:
        return 10

    def foo(_rnd, _seed):
        if False:
            while True:
                i = 10
        try:
            return (_rnd.integers(0, 1000), _seed)
        except Exception:
            return (_rnd.randint(0, 1000), _seed)
    cf = create_captured_function(foo)
    assert cf.uses_randomness
    cf.logger = mock.MagicMock()
    cf.rnd = random.Random(1234)
    (nr1, seed1) = cf()
    (nr2, seed2) = cf()
    assert nr1 != nr2
    assert seed1 != seed2
    cf.rnd = random.Random(1234)
    assert cf() == (nr1, seed1)
    assert cf() == (nr2, seed2)

def test_captured_function_numpy_randomness():
    if False:
        i = 10
        return i + 15

    def foo(_rnd, _seed):
        if False:
            print('Hello World!')
        return (_rnd, _seed)
    cf = create_captured_function(foo)
    assert cf.uses_randomness
    cf.logger = mock.MagicMock()
    cf.rnd = random.Random(1234)
    SETTINGS.CONFIG.NUMPY_RANDOM_LEGACY_API = False
    (rnd, seed) = cf()
    if opt.has_numpy:
        assert type(rnd) == opt.np.random.Generator
        SETTINGS.CONFIG.NUMPY_RANDOM_LEGACY_API = True
        (rnd, seed) = cf()
        assert type(rnd) == opt.np.random.RandomState
    else:
        assert type(rnd) == random.Random

def test_captured_function_magic_logger_argument():
    if False:
        i = 10
        return i + 15

    def foo(_log):
        if False:
            while True:
                i = 10
        return _log
    cf = create_captured_function(foo)
    cf.logger = mock.MagicMock()
    assert cf() == cf.logger

def test_captured_function_magic_config_argument():
    if False:
        while True:
            i = 10

    def foo(_config):
        if False:
            while True:
                i = 10
        return _config
    cf = create_captured_function(foo)
    cf.logger = mock.MagicMock()
    cf.config = {'a': 2, 'b': 2}
    assert cf() == cf.config

def test_captured_function_magic_run_argument():
    if False:
        return 10

    def foo(_run):
        if False:
            print('Hello World!')
        return _run
    cf = create_captured_function(foo)
    cf.logger = mock.MagicMock()
    cf.run = mock.MagicMock()
    assert cf() == cf.run

def test_captured_function_call_doesnt_modify_kwargs():
    if False:
        print('Hello World!')

    def foo(a, _log):
        if False:
            while True:
                i = 10
        if _log is not None:
            return a
    cf = create_captured_function(foo)
    cf.logger = mock.MagicMock()
    cf.run = mock.MagicMock()
    d = {'a': 7}
    assert cf(**d) == 7
    assert d == {'a': 7}