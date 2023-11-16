""" Reference counting tests for features of Python3.3 or higher.

These contain functions that do specific things, where we have a suspect
that references may be lost or corrupted. Executing them repeatedly and
checking the reference count is how they are used.

These are Python3 specific constructs, that will give a SyntaxError or
not be relevant on Python2.
"""
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import types
from nuitka.tools.testing.Common import executeReferenceChecked, someGenerator, someGeneratorRaising

def simpleFunction1():
    if False:
        return 10

    def abc(*, _exc=IOError):
        if False:
            i = 10
            return i + 15
        pass
    for _ in range(100):
        abc()

def simpleFunction2():
    if False:
        print('Hello World!')

    def abc(*, exc=IOError):
        if False:
            return 10
        raise ValueError from None
    try:
        abc()
    except (ValueError, TypeError):
        pass

def simpleFunction3():
    if False:
        for i in range(10):
            print('nop')
    try:

        class ClassA(Exception):
            pass

        class ClassB(Exception):
            pass
        try:
            raise ClassA('foo')
        except ClassA as e1:
            raise ClassB(str(e1)) from e1
    except Exception:
        pass

def simpleFunction4():
    if False:
        i = 10
        return i + 15
    a = 1

    def nonlocal_writer():
        if False:
            i = 10
            return i + 15
        nonlocal a
        for a in range(10):
            pass
    nonlocal_writer()
    assert a == 9, a

def simpleFunction5():
    if False:
        return 10
    x = 2

    def local_func(_a: int, _b: x * x):
        if False:
            while True:
                i = 10
        pass
    local_func(x, x)

def simpleFunction6():
    if False:
        for i in range(10):
            print('nop')

    class MyException(Exception):

        def __init__(self, obj):
            if False:
                print('Hello World!')
            self.obj = obj

    class MyObj:
        pass

    def inner_raising_func():
        if False:
            return 10
        local_ref = obj
        raise MyException(obj)
    obj = MyObj()
    try:
        try:
            inner_raising_func()
        except:
            raise KeyError
    except KeyError as e:
        pass
range_low = 0
range_high = 256
range_step = 13

def simpleFunction7():
    if False:
        i = 10
        return i + 15
    return range(range_low, range_high, range_step)

def simpleFunction8():
    if False:
        i = 10
        return i + 15
    return range(range_low, range_high)

def simpleFunction9():
    if False:
        while True:
            i = 10
    return range(range_high)

def simpleFunction10():
    if False:
        return 10

    def f(_x: int) -> int:
        if False:
            while True:
                i = 10
        pass
    return f

def simpleFunction11():
    if False:
        for i in range(10):
            print('nop')
    try:
        raise ImportError(path='lala', name='lele')
    except ImportError as e:
        assert e.name == 'lele'
        assert e.path == 'lala'

def simpleFunction12():
    if False:
        while True:
            i = 10

    def g():
        if False:
            print('Hello World!')
        for a in range(20):
            yield a

    def h():
        if False:
            return 10
        yield 4
        yield 5
        yield 6

    def f():
        if False:
            while True:
                i = 10
        yield from g()
        yield from h()
    _x = list(f())

def simpleFunction13():
    if False:
        print('Hello World!')

    def g():
        if False:
            return 10
        for a in range(20):
            yield a

    def h():
        if False:
            for i in range(10):
                print('nop')
        yield 4
        yield 5
        yield 6
        raise TypeError

    def f():
        if False:
            while True:
                i = 10
        yield from g()
        yield from h()
    try:
        _x = list(f())
    except TypeError:
        pass

class Broken:

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            return 10
        return 1

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        1 / 0

def simpleFunction14():
    if False:
        for i in range(10):
            print('nop')

    def g():
        if False:
            i = 10
            return i + 15
        yield from Broken()
    try:
        gi = g()
        next(gi)
    except Exception:
        pass

def simpleFunction15():
    if False:
        print('Hello World!')

    def g():
        if False:
            i = 10
            return i + 15
        yield from Broken()
    try:
        gi = g()
        next(gi)
        gi.throw(AttributeError)
    except Exception:
        pass

def simpleFunction16():
    if False:
        while True:
            i = 10

    def g():
        if False:
            print('Hello World!')
        yield from (2, 3)
    return list(g())

def simpleFunction17():
    if False:
        for i in range(10):
            print('nop')

    def g():
        if False:
            i = 10
            return i + 15
        yield from (2, 3)
        return 9
    return list(g())

def simpleFunction18():
    if False:
        return 10

    def g():
        if False:
            return 10
        yield from (2, 3)
        return (9, 8)
    return list(g())

def simpleFunction19():
    if False:
        return 10

    def g():
        if False:
            for i in range(10):
                print('nop')
        x = someGenerator()
        assert type(x) is types.GeneratorType
        yield from x
    gen = g()
    next(gen)
    try:
        gen.throw(ValueError)
    except ValueError:
        pass

def simpleFunction20():
    if False:
        print('Hello World!')

    def g():
        if False:
            for i in range(10):
                print('nop')
        x = someGeneratorRaising()
        assert type(x) is types.GeneratorType
        yield from x
    gen = g()
    next(gen)
    try:
        next(gen)
    except TypeError:
        pass

class ClassIteratorBrokenClose:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.my_iter = iter(range(2))

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def next(self):
        if False:
            return 10
        return next(self.my_iter)

    def close(self):
        if False:
            while True:
                i = 10
        raise TypeError(3)
    __next__ = next

def simpleFunction21():
    if False:
        print('Hello World!')

    def g():
        if False:
            for i in range(10):
                print('nop')
        x = ClassIteratorBrokenClose()
        yield from x
    gen = g()
    next(gen)
    try:
        gen.throw(GeneratorExit)
    except TypeError:
        pass

class ClassIteratorBrokenThrow:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.my_iter = iter(range(2))

    def __iter__(self):
        if False:
            return 10
        return self

    def next(self):
        if False:
            while True:
                i = 10
        return next(self.my_iter)

    def throw(self, *args):
        if False:
            while True:
                i = 10
        raise TypeError(3)
    __next__ = next

def simpleFunction22():
    if False:
        i = 10
        return i + 15

    def g():
        if False:
            for i in range(10):
                print('nop')
        x = ClassIteratorBrokenThrow()
        yield from x
    gen = g()
    next(gen)
    try:
        gen.throw(ValueError)
    except GeneratorExit:
        pass
    except TypeError:
        pass

class ClassIteratorRejectingThrow:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.my_iter = iter(range(2))

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def next(self):
        if False:
            return 10
        return next(self.my_iter)

    def throw(self, *args):
        if False:
            print('Hello World!')
        assert len(args) == 1, args
    __next__ = next

class MyError(Exception):

    def __init__(self):
        if False:
            return 10
        assert False

def simpleFunction23():
    if False:
        while True:
            i = 10

    def g():
        if False:
            i = 10
            return i + 15
        x = ClassIteratorRejectingThrow()
        yield from x
    gen = g()
    next(gen)
    gen.throw(MyError)
oho = 1

def simpleFunction24():
    if False:
        for i in range(10):
            print('nop')

    def someGenerator():
        if False:
            for i in range(10):
                print('nop')
        yield from oho
    try:
        list(someGenerator())
    except TypeError:
        pass
tests_stderr = (14, 15)
tests_skipped = {}
result = executeReferenceChecked(prefix='simpleFunction', names=globals(), tests_skipped=tests_skipped, tests_stderr=tests_stderr)
sys.exit(0 if result else 1)