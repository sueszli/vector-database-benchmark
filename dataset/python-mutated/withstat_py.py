def typename(t):
    if False:
        return 10
    name = type(t).__name__
    return "<type '%s'>" % name

class MyException(Exception):
    pass

class ContextManager(object):

    def __init__(self, value, exit_ret=None):
        if False:
            print('Hello World!')
        self.value = value
        self.exit_ret = exit_ret

    def __exit__(self, a, b, tb):
        if False:
            print('Hello World!')
        print('exit %s %s %s' % (typename(a), typename(b), typename(tb)))
        return self.exit_ret

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        print('enter')
        return self.value

def no_as():
    if False:
        i = 10
        return i + 15
    "\n    >>> no_as()\n    enter\n    hello\n    exit <type 'NoneType'> <type 'NoneType'> <type 'NoneType'>\n    "
    with ContextManager('value'):
        print('hello')

def basic():
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> basic()\n    enter\n    value\n    exit <type 'NoneType'> <type 'NoneType'> <type 'NoneType'>\n    "
    with ContextManager('value') as x:
        print(x)

def with_pass():
    if False:
        i = 10
        return i + 15
    "\n    >>> with_pass()\n    enter\n    exit <type 'NoneType'> <type 'NoneType'> <type 'NoneType'>\n    "
    with ContextManager('value') as x:
        pass

def with_return():
    if False:
        print('Hello World!')
    "\n    >>> print(with_return())\n    enter\n    exit <type 'NoneType'> <type 'NoneType'> <type 'NoneType'>\n    value\n    "
    with ContextManager('value') as x:
        return x

def with_break():
    if False:
        print('Hello World!')
    "\n    >>> print(with_break())\n    enter\n    exit <type 'NoneType'> <type 'NoneType'> <type 'NoneType'>\n    a\n    "
    for c in list('abc'):
        with ContextManager('value') as x:
            break
        print('FAILED')
    return c

def with_continue():
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> print(with_continue())\n    enter\n    exit <type 'NoneType'> <type 'NoneType'> <type 'NoneType'>\n    enter\n    exit <type 'NoneType'> <type 'NoneType'> <type 'NoneType'>\n    enter\n    exit <type 'NoneType'> <type 'NoneType'> <type 'NoneType'>\n    c\n    "
    for c in list('abc'):
        with ContextManager('value') as x:
            continue
        print('FAILED')
    return c

def with_exception(exit_ret):
    if False:
        print('Hello World!')
    "\n    >>> with_exception(None)\n    enter\n    value\n    exit <type 'type'> <type 'MyException'> <type 'traceback'>\n    outer except\n    >>> with_exception(True)\n    enter\n    value\n    exit <type 'type'> <type 'MyException'> <type 'traceback'>\n    "
    try:
        with ContextManager('value', exit_ret=exit_ret) as value:
            print(value)
            raise MyException()
    except:
        print('outer except')

def with_real_lock():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> with_real_lock()\n    about to acquire lock\n    holding lock\n    lock no longer held\n    '
    from threading import Lock
    lock = Lock()
    print('about to acquire lock')
    with lock:
        print('holding lock')
    print('lock no longer held')

def functions_in_with():
    if False:
        while True:
            i = 10
    "\n    >>> f = functions_in_with()\n    enter\n    exit <type 'type'> <type 'MyException'> <type 'traceback'>\n    outer except\n    >>> f(1)[0]\n    1\n    >>> print(f(1)[1])\n    value\n    "
    try:
        with ContextManager('value') as value:

            def f(x):
                if False:
                    print('Hello World!')
                return (x, value)
            make = lambda x: x()
            raise make(MyException)
    except:
        print('outer except')
    return f

def multitarget():
    if False:
        i = 10
        return i + 15
    "\n    >>> multitarget()\n    enter\n    1 2 3 4 5\n    exit <type 'NoneType'> <type 'NoneType'> <type 'NoneType'>\n    "
    with ContextManager((1, 2, (3, (4, 5)))) as (a, b, (c, (d, e))):
        print('%s %s %s %s %s' % (a, b, c, d, e))

def tupletarget():
    if False:
        return 10
    "\n    >>> tupletarget()\n    enter\n    (1, 2, (3, (4, 5)))\n    exit <type 'NoneType'> <type 'NoneType'> <type 'NoneType'>\n    "
    with ContextManager((1, 2, (3, (4, 5)))) as t:
        print(t)

class GetManager(object):

    def get(self, *args):
        if False:
            print('Hello World!')
        return ContextManager(*args)

def manager_from_expression():
    if False:
        i = 10
        return i + 15
    "\n    >>> manager_from_expression()\n    enter\n    1\n    exit <type 'NoneType'> <type 'NoneType'> <type 'NoneType'>\n    enter\n    2\n    exit <type 'NoneType'> <type 'NoneType'> <type 'NoneType'>\n    "
    with GetManager().get(1) as x:
        print(x)
    g = GetManager()
    with g.get(2) as x:
        print(x)

def manager_from_ternary(use_first):
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> manager_from_ternary(True)\n    enter\n    exit <type 'type'> <type 'ValueError'> <type 'traceback'>\n    >>> manager_from_ternary(False)\n    enter\n    exit <type 'type'> <type 'ValueError'> <type 'traceback'>\n    In except\n    "
    cm1_getter = lambda : ContextManager('1', exit_ret=True)
    cm2_getter = lambda : ContextManager('2')
    try:
        with (cm1_getter if use_first else cm2_getter)():
            raise ValueError
    except ValueError:
        print('In except')