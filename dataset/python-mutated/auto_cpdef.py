import cython

def str(arg):
    if False:
        while True:
            i = 10
    "\n    This is a bit evil - str gets mapped to a C-API function and is\n    being redefined here.\n\n    >>> print(str('TEST'))\n    STR\n    "
    return 'STR'

@cython.test_assert_path_exists('//SimpleCallNode[@function.type.is_cfunction = True]')
@cython.test_fail_if_path_exists('//SimpleCallNode[@function.type.is_builtin_type = True]')
def call_str(arg):
    if False:
        print('Hello World!')
    "\n    >>> print(call_str('TEST'))\n    STR\n    "
    return str(arg)

def stararg_func(*args):
    if False:
        while True:
            i = 10
    '\n    >>> stararg_func(1, 2)\n    (1, 2)\n    '
    return args

def starstararg_func(**kwargs):
    if False:
        while True:
            i = 10
    '\n    >>> starstararg_func(a=1)\n    1\n    '
    return kwargs['a']
l = lambda x: 1

def test_lambda():
    if False:
        i = 10
        return i + 15
    '\n    >>> l(1)\n    1\n    '
try:
    from math import fabs
except ImportError:

    def fabs(x):
        if False:
            for i in range(10):
                print('nop')
        if x < 0:
            return -x
        else:
            return x
try:
    from math import no_such_function
except ImportError:

    def no_such_function(x):
        if False:
            for i in range(10):
                print('nop')
        return x + 1.0

def test_import_fallback():
    if False:
        while True:
            i = 10
    '\n    >>> fabs(1.0)\n    1.0\n    >>> no_such_function(1.0)\n    2.0\n    >>> test_import_fallback()\n    (1.0, 2.0)\n    '
    return (fabs(1.0), no_such_function(1.0))