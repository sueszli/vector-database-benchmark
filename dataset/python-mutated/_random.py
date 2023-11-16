import atexit
import functools
import numpy
import os
import random
import types
import unittest
import cupy
_old_python_random_state = None
_old_numpy_random_state = None
_old_cupy_random_states = None

def do_setup(deterministic=True):
    if False:
        while True:
            i = 10
    global _old_python_random_state
    global _old_numpy_random_state
    global _old_cupy_random_states
    _old_python_random_state = random.getstate()
    _old_numpy_random_state = numpy.random.get_state()
    _old_cupy_random_states = cupy.random._generator._random_states
    cupy.random.reset_states()
    assert cupy.random._generator._random_states is not _old_cupy_random_states
    if not deterministic:
        random.seed()
        numpy.random.seed()
        cupy.random.seed()
    else:
        random.seed(99)
        numpy.random.seed(100)
        cupy.random.seed(101)

def do_teardown():
    if False:
        while True:
            i = 10
    global _old_python_random_state
    global _old_numpy_random_state
    global _old_cupy_random_states
    random.setstate(_old_python_random_state)
    numpy.random.set_state(_old_numpy_random_state)
    cupy.random._generator._random_states = _old_cupy_random_states
    _old_python_random_state = None
    _old_numpy_random_state = None
    _old_cupy_random_states = None
_nest_count = 0

@atexit.register
def _check_teardown():
    if False:
        return 10
    assert _nest_count == 0, '_setup_random() and _teardown_random() must be called in pairs.'

def _setup_random():
    if False:
        i = 10
        return i + 15
    'Sets up the deterministic random states of ``numpy`` and ``cupy``.\n\n    '
    global _nest_count
    if _nest_count == 0:
        nondeterministic = bool(int(os.environ.get('CUPY_TEST_RANDOM_NONDETERMINISTIC', '0')))
        do_setup(not nondeterministic)
    _nest_count += 1

def _teardown_random():
    if False:
        while True:
            i = 10
    'Tears down the deterministic random states set up by ``_setup_random``.\n\n    '
    global _nest_count
    assert _nest_count > 0, '_setup_random has not been called'
    _nest_count -= 1
    if _nest_count == 0:
        do_teardown()

def generate_seed():
    if False:
        while True:
            i = 10
    assert _nest_count > 0, 'random is not set up'
    return numpy.random.randint(2147483647)

def fix_random():
    if False:
        while True:
            i = 10
    'Decorator that fixes random numbers in a test.\n\n    This decorator can be applied to either a test case class or a test method.\n    It should not be applied within ``condition.retry`` or\n    ``condition.repeat``.\n    '

    def decorator(impl):
        if False:
            i = 10
            return i + 15
        if isinstance(impl, types.FunctionType) and impl.__name__.startswith('test_'):

            @functools.wraps(impl)
            def test_func(self, *args, **kw):
                if False:
                    i = 10
                    return i + 15
                _setup_random()
                try:
                    impl(self, *args, **kw)
                finally:
                    _teardown_random()
            return test_func
        elif isinstance(impl, type) and issubclass(impl, unittest.TestCase):
            klass = impl

            def wrap_setUp(f):
                if False:
                    for i in range(10):
                        print('nop')

                def func(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    _setup_random()
                    f(self)
                return func

            def wrap_tearDown(f):
                if False:
                    for i in range(10):
                        print('nop')

                def func(self):
                    if False:
                        return 10
                    try:
                        f(self)
                    finally:
                        _teardown_random()
                return func
            klass.setUp = wrap_setUp(klass.setUp)
            klass.tearDown = wrap_tearDown(klass.tearDown)
            return klass
        else:
            raise ValueError("Can't apply fix_random to {}".format(impl))
    return decorator