import sys
import cython

def very_simple():
    if False:
        i = 10
        return i + 15
    "\n    >>> x = very_simple()\n    >>> next(x)\n    1\n    >>> next(x)\n    Traceback (most recent call last):\n    StopIteration\n    >>> next(x)\n    Traceback (most recent call last):\n    StopIteration\n    >>> x = very_simple()\n    >>> x.send(1)\n    Traceback (most recent call last):\n    TypeError: can't send non-None value to a just-started generator\n    "
    yield 1

def simple():
    if False:
        return 10
    '\n    >>> x = simple()\n    >>> list(x)\n    [1, 2, 3]\n    '
    yield 1
    yield 2
    yield 3

def simple_seq(seq):
    if False:
        return 10
    '\n    >>> x = simple_seq("abc")\n    >>> list(x)\n    [\'a\', \'b\', \'c\']\n    '
    for i in seq:
        yield i

def simple_send():
    if False:
        while True:
            i = 10
    '\n    >>> x = simple_send()\n    >>> next(x)\n    >>> x.send(1)\n    1\n    >>> x.send(2)\n    2\n    >>> x.send(3)\n    3\n    '
    i = None
    while True:
        i = (yield i)

def raising():
    if False:
        return 10
    "\n    >>> x = raising()\n    >>> next(x)\n    Traceback (most recent call last):\n    KeyError: 'foo'\n    >>> next(x)\n    Traceback (most recent call last):\n    StopIteration\n    "
    yield {}['foo']

def with_outer(*args):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> x = with_outer(1, 2, 3)\n    >>> list(x())\n    [1, 2, 3]\n    '

    def generator():
        if False:
            i = 10
            return i + 15
        for i in args:
            yield i
    return generator

def test_close():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> x = test_close()\n    >>> x.close()\n    >>> x = test_close()\n    >>> next(x)\n    >>> x.close()\n    >>> next(x)\n    Traceback (most recent call last):\n    StopIteration\n    '
    while True:
        yield

def test_ignore_close():
    if False:
        i = 10
        return i + 15
    '\n    >>> x = test_ignore_close()\n    >>> x.close()\n    >>> x = test_ignore_close()\n    >>> next(x)\n    >>> x.close()\n    Traceback (most recent call last):\n    RuntimeError: generator ignored GeneratorExit\n    '
    try:
        yield
    except GeneratorExit:
        yield

def check_throw():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> x = check_throw()\n    >>> x.throw(ValueError)\n    Traceback (most recent call last):\n    ValueError\n    >>> next(x)\n    Traceback (most recent call last):\n    StopIteration\n    >>> x = check_throw()\n    >>> next(x)\n    >>> x.throw(ValueError)\n    >>> next(x)\n    >>> x.throw(IndexError, "oops")\n    Traceback (most recent call last):\n    IndexError: oops\n    >>> next(x)\n    Traceback (most recent call last):\n    StopIteration\n    '
    while True:
        try:
            yield
        except ValueError:
            pass

def check_yield_in_except():
    if False:
        while True:
            i = 10
    '\n    >>> try:\n    ...     raise TypeError("RAISED !")\n    ... except TypeError as orig_exc:\n    ...     assert isinstance(orig_exc, TypeError), orig_exc\n    ...     g = check_yield_in_except()\n    ...     print(orig_exc is sys.exc_info()[1] or sys.exc_info())\n    ...     next(g)\n    ...     print(orig_exc is sys.exc_info()[1] or sys.exc_info())\n    ...     next(g)\n    ...     print(orig_exc is sys.exc_info()[1] or sys.exc_info())\n    True\n    True\n    True\n    >>> next(g)\n    Traceback (most recent call last):\n    StopIteration\n    '
    try:
        yield
        raise ValueError
    except ValueError as exc:
        assert sys.exc_info()[1] is exc, sys.exc_info()
        yield
        assert sys.exc_info()[1] is exc, sys.exc_info()

def yield_in_except_throw_exc_type():
    if False:
        while True:
            i = 10
    '\n    >>> g = yield_in_except_throw_exc_type()\n    >>> next(g)\n    >>> g.throw(TypeError)\n    Traceback (most recent call last):\n    TypeError\n    >>> next(g)\n    Traceback (most recent call last):\n    StopIteration\n    '
    try:
        raise ValueError
    except ValueError as exc:
        assert sys.exc_info()[1] is exc, sys.exc_info()
        yield
        assert sys.exc_info()[1] is exc, sys.exc_info()

def yield_in_except_throw_instance():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> g = yield_in_except_throw_instance()\n    >>> next(g)\n    >>> g.throw(TypeError())\n    Traceback (most recent call last):\n    TypeError\n    >>> next(g)\n    Traceback (most recent call last):\n    StopIteration\n    '
    try:
        raise ValueError
    except ValueError as exc:
        assert sys.exc_info()[1] is exc, sys.exc_info()
        yield
        assert sys.exc_info()[1] is exc, sys.exc_info()

def test_swap_assignment():
    if False:
        while True:
            i = 10
    '\n    >>> gen = test_swap_assignment()\n    >>> next(gen)\n    (5, 10)\n    >>> next(gen)\n    (10, 5)\n    '
    (x, y) = (5, 10)
    yield (x, y)
    (x, y) = (y, x)
    yield (x, y)

class Foo(object):
    """
    >>> obj = Foo()
    >>> list(obj.simple(1, 2, 3))
    [1, 2, 3]
    """

    def simple(self, *args):
        if False:
            i = 10
            return i + 15
        for i in args:
            yield i

def test_nested(a, b, c):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> obj = test_nested(1, 2, 3)\n    >>> [i() for i in obj]\n    [1, 2, 3, 4]\n    '

    def one():
        if False:
            return 10
        return a

    def two():
        if False:
            i = 10
            return i + 15
        return b

    def three():
        if False:
            return 10
        return c

    def new_closure(a, b):
        if False:
            i = 10
            return i + 15

        def sum():
            if False:
                for i in range(10):
                    print('nop')
            return a + b
        return sum
    yield one
    yield two
    yield three
    yield new_closure(a, c)

def tolist(func):
    if False:
        for i in range(10):
            print('nop')

    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return list(func(*args, **kwargs))
    return wrapper

@tolist
def test_decorated(*args):
    if False:
        while True:
            i = 10
    '\n    >>> test_decorated(1, 2, 3)\n    [1, 2, 3]\n    '
    for i in args:
        yield i

def test_return(a):
    if False:
        print('Hello World!')
    "\n    >>> d = dict()\n    >>> obj = test_return(d)\n    >>> next(obj)\n    1\n    >>> next(obj)\n    Traceback (most recent call last):\n    StopIteration\n    >>> d['i_was_here']\n    True\n    "
    yield 1
    a['i_was_here'] = True
    return

def test_copied_yield(foo):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> class Manager(object):\n    ...    def __enter__(self):\n    ...        return self\n    ...    def __exit__(self, type, value, tb):\n    ...        pass\n    >>> list(test_copied_yield(Manager()))\n    [1]\n    '
    with foo:
        yield 1

def test_nested_yield():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> obj = test_nested_yield()\n    >>> next(obj)\n    1\n    >>> obj.send(2)\n    2\n    >>> obj.send(3)\n    3\n    >>> obj.send(4)\n    Traceback (most recent call last):\n    StopIteration\n    '
    yield (yield (yield 1))

def test_sum_of_yields(n):
    if False:
        return 10
    '\n    >>> g = test_sum_of_yields(3)\n    >>> next(g)\n    (0, 0)\n    >>> g.send(1)\n    (0, 1)\n    >>> g.send(1)\n    (1, 2)\n    '
    x = 0
    x += (yield (0, x))
    x += (yield (0, x))
    yield (1, x)

def test_nested_gen(n):
    if False:
        i = 10
        return i + 15
    '\n    >>> [list(a) for a in test_nested_gen(5)]\n    [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]\n    '
    for a in range(n):
        yield (b for b in range(a))

def test_lambda(n):
    if False:
        print('Hello World!')
    '\n    >>> [i() for i in test_lambda(3)]\n    [0, 1, 2]\n    '
    for i in range(n):
        yield (lambda : i)

def test_generator_cleanup():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> g = test_generator_cleanup()\n    >>> del g\n    >>> g = test_generator_cleanup()\n    >>> next(g)\n    1\n    >>> del g\n    cleanup\n    '
    try:
        yield 1
    finally:
        print('cleanup')

def test_del_in_generator():
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> [ s for s in test_del_in_generator() ]\n    ['abcabcabc', 'abcabcabc']\n    "
    x = len('abc') * 'abc'
    a = x
    yield x
    del x
    yield a
    del a

@cython.test_fail_if_path_exists('//IfStatNode', '//PrintStatNode')
def test_yield_in_const_conditional_false():
    if False:
        return 10
    '\n    >>> list(test_yield_in_const_conditional_false())\n    []\n    '
    if False:
        print((yield 1))

@cython.test_fail_if_path_exists('//IfStatNode')
@cython.test_assert_path_exists('//PrintStatNode')
def test_yield_in_const_conditional_true():
    if False:
        while True:
            i = 10
    '\n    >>> list(test_yield_in_const_conditional_true())\n    None\n    [1]\n    '
    if True:
        print((yield 1))

def test_generator_scope():
    if False:
        while True:
            i = 10
    "\n    Tests that the function is run at the correct time\n    (i.e. when the generator is created, not when it's run)\n    >>> list(test_generator_scope())\n    inner running\n    generator created\n    [0, 10]\n    "

    def inner(val):
        if False:
            i = 10
            return i + 15
        print('inner running')
        return [0, val]
    gen = (a for a in inner(10))
    print('generator created')
    return gen