def test1(args_1, c: int, w=4, *varargs: int, **kwargs: 'annotating kwargs') -> tuple:
    if False:
        return 10
    return (args_1, c, w, kwargs)

def test2(args_1, args_2, c: int, w=4, *varargs: int, **kwargs: 'annotating kwargs'):
    if False:
        i = 10
        return i + 15
    return (args_1, args_2, c, w, varargs, kwargs)

def test3(c: int, w=4, *varargs: int, **kwargs: 'annotating kwargs') -> float:
    if False:
        for i in range(10):
            print('nop')
    return 5.4

def test4(a: float, c: int, *varargs: int, **kwargs: 'annotating kwargs') -> float:
    if False:
        print('Hello World!')
    return 5.4

def test5(a: float, c: int=5, *varargs: int, **kwargs: 'annotating kwargs') -> float:
    if False:
        print('Hello World!')
    return 5.4

def test6(a: float, c: int, test=None):
    if False:
        while True:
            i = 10
    return (a, c, test)

def test7(*varargs: int, **kwargs):
    if False:
        i = 10
        return i + 15
    return (varargs, kwargs)

def test8(x=55, *varargs: int, **kwargs) -> list:
    if False:
        for i in range(10):
            print('nop')
    return (x, varargs, kwargs)

def test9(arg_1=55, *varargs: int, y=5, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return (x, varargs, int, y, kwargs)

def test10(args_1, b: 'annotating b', c: int) -> float:
    if False:
        while True:
            i = 10
    return 5.4

def test11(*, name):
    if False:
        print('Hello World!')
    return (args, name)

def test12(a, *args, name):
    if False:
        i = 10
        return i + 15
    return (a, args)
    pass

def test13(*args, name):
    if False:
        return 10
    return (args, name)

def test14(*args, name: int=1, qname):
    if False:
        while True:
            i = 10
    return (args, name, qname)

def test15(*args, name='S', fname, qname=4):
    if False:
        while True:
            i = 10
    return (args, name, fname, qname)
_DEFAULT_LIMIT = 5

def test16(host=None, port=None, *, loop=None, limit=_DEFAULT_LIMIT, **kwds):
    if False:
        i = 10
        return i + 15
    return (host, port, loop, limit, kwds)

def o(f, mode='r', buffering=None) -> 'IOBase':
    if False:
        while True:
            i = 10
    return (f, mode, buffering)

def foo1(x: 'an argument that defaults to 5'=5):
    if False:
        while True:
            i = 10
    print(x)

def div(a: dict(type=float, help='the dividend'), b: dict(type=float, help='the divisor (must be different than 0)')) -> dict(type=float, help='the result of dividing a by b'):
    if False:
        for i in range(10):
            print('nop')
    'Divide a by b'
    return a / b

def f(a: 'This is a new annotation'):
    if False:
        print('Hello World!')
    'This is a test'
    assert f.__annotations__['a'] == 'This is a new annotation'
f(5)

class TestSignatureObject1:

    def test_signature_on_wkwonly(self):
        if False:
            return 10

        def test(*, a: float, b: str, c: str='test', **kwargs: int) -> int:
            if False:
                print('Hello World!')
            pass

class TestSignatureObject2:

    def test_signature_on_wkwonly(self):
        if False:
            i = 10
            return i + 15

        def test(*, c='test', a: float, b: str='S', **kwargs: int) -> int:
            if False:
                while True:
                    i = 10
            pass

class TestSignatureObject3:

    def test_signature_on_wkwonly(self):
        if False:
            while True:
                i = 10

        def test(*, c='test', a: float, kwargs: str='S', **b: int) -> int:
            if False:
                return 10
            pass

class TestSignatureObject4:

    def test_signature_on_wkwonly(self):
        if False:
            while True:
                i = 10

        def test(x=55, *args, c: str='test', a: float, kwargs: str='S', **b: int) -> int:
            if False:
                while True:
                    i = 10
            pass

class TestSignatureObject5:

    def test_signature_on_wkwonly(self):
        if False:
            for i in range(10):
                print('nop')

        def test(x=55, *args: int, c='test', a: float, kwargs: str='S', **b: int) -> int:
            if False:
                return 10
            pass

class TestSignatureObject5:

    def test_signature_on_wkwonly(self):
        if False:
            while True:
                i = 10

        def test(x: int=55, *args: (int, str), c='test', a: float, kwargs: str='S', **b: int) -> int:
            if False:
                print('Hello World!')
            pass

class TestSignatureObject7:

    def test_signature_on_wkwonly(self):
        if False:
            for i in range(10):
                print('nop')

        def test(c='test', kwargs: str='S', **b: int) -> int:
            if False:
                return 10
            pass

class TestSignatureObject8:

    def test_signature_on_wkwonly(self):
        if False:
            while True:
                i = 10

        def test(**b: int) -> int:
            if False:
                for i in range(10):
                    print('nop')
            pass

class TestSignatureObject9:

    def test_signature_on_wkwonly(self):
        if False:
            i = 10
            return i + 15

        def test(a, **b: int) -> int:
            if False:
                while True:
                    i = 10
            pass

class SupportsInt:

    def __int__(self) -> int:
        if False:
            i = 10
            return i + 15
        pass

def ann1(args_1, b: 'annotating b', c: int, *varargs: str) -> float:
    if False:
        while True:
            i = 10
    assert ann1.__annotations__['b'] == 'annotating b'
    assert ann1.__annotations__['c'] == int
    assert ann1.__annotations__['varargs'] == str
    assert ann1.__annotations__['return'] == float

def ann2(args_1, b: int=5, **kwargs: float) -> float:
    if False:
        while True:
            i = 10
    assert ann2.__annotations__['b'] == int
    assert ann2.__annotations__['kwargs'] == float
    assert ann2.__annotations__['return'] == float
    assert b == 5

class TestSignatureObject:

    def test_signature_on_wkwonly(self):
        if False:
            return 10

        def test(x: int=55, *args: (int, str), c='test', a: float, kwargs: str='S', **b: int) -> int:
            if False:
                print('Hello World!')
            pass
assert test1(1, 5) == (1, 5, 4, {})
assert test1(1, 5, 6, foo='bar') == (1, 5, 6, {'foo': 'bar'})
assert test2(2, 3, 4) == (2, 3, 4, 4, (), {})
assert test3(10, foo='bar') == 5.4
assert test4(9.5, 7, 6, 4, bar='baz') == 5.4
assert test6(1.2, 3) == (1.2, 3, None)
assert test6(2.3, 4, 5) == (2.3, 4, 5)
ann1(1, 'test', 5)
ann2(1)
assert test12(1, 2, 3, name='hi') == (1, (2, 3)), 'a, *args, name'
assert test13(1, 2, 3, name='hi') == ((1, 2, 3), 'hi'), '*args, name'
assert test16('localhost', loop=2, limit=3, a='b') == ('localhost', None, 2, 3, {'a': 'b'})
try:
    import typing

    def foo() -> typing.Iterator[typing.Tuple[int, typing.Any]]:
        if False:
            i = 10
            return i + 15
        ...
except:
    pass