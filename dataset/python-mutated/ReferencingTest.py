""" Reference counting tests.

These contain functions that do specific things, where we have a suspect
that references may be lost or corrupted. Executing them repeatedly and
checking the reference count is how they are used.
"""
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from nuitka.tools.testing.Common import executeReferenceChecked
x = 17

def simpleFunction1():
    if False:
        i = 10
        return i + 15
    return 1

def simpleFunction2():
    if False:
        i = 10
        return i + 15
    y = 3 * x
    y = 3
    return x * 2 * y

def simpleFunction3():
    if False:
        print('Hello World!')

    def contained():
        if False:
            return 10
        return x
    return contained

def simpleFunction4(a=1):
    if False:
        while True:
            i = 10
    y = a

    def contained():
        if False:
            for i in range(10):
                print('nop')
        return y
    return contained

def simpleFunction5(a=2):
    if False:
        return 10
    c = 1
    f = [a, a + c]
    return (c, f)

def simpleFunction6():
    if False:
        print('Hello World!')
    for _b in range(6):
        pass
    for _c in (1, 2, 3, 4, 5, 6):
        pass

def simpleFunction7(b=1):
    if False:
        print('Hello World!')
    for _b in range(6):
        pass

def simpleFunction8():
    if False:
        return 10
    c = []
    c.append(x)

def simpleFunction9(a=1 * 2):
    if False:
        while True:
            i = 10
    if a == a:
        pass
u = None

def simpleFunction10(a=1 * 2):
    if False:
        return 10
    x = [u for u in range(8)]

def simpleFunction11():
    if False:
        while True:
            i = 10
    f = 1
    while f < 8:
        f += 1
v = None

def simpleFunction12():
    if False:
        for i in range(10):
            print('nop')
    a = [(u, v) for (u, v) in zip(range(8), range(8))]
    return a

def cond():
    if False:
        i = 10
        return i + 15
    return 1

def simpleFunction13(a=1 * 2):
    if False:
        while True:
            i = 10
    pass

def simpleFunction14p(x):
    if False:
        while True:
            i = 10
    try:
        simpleFunction14p(1, 1)
    except TypeError as _e:
        pass
    try:
        simpleFunction14p(1, 1)
    except TypeError:
        pass

def simpleFunction14():
    if False:
        i = 10
        return i + 15
    simpleFunction14p(3)

def simpleFunction15p(x):
    if False:
        while True:
            i = 10
    try:
        try:
            x += 1
        finally:
            try:
                x *= 1
            finally:
                _z = 1
    except:
        pass

def simpleFunction15():
    if False:
        for i in range(10):
            print('nop')
    simpleFunction15p([1])

def simpleFunction16():
    if False:
        i = 10
        return i + 15

    class EmptyClass:
        pass
    return EmptyClass

def simpleFunction17():
    if False:
        print('Hello World!')

    class EmptyObjectClass:
        pass
    return EmptyObjectClass()

def simpleFunction18():
    if False:
        for i in range(10):
            print('nop')
    closured = 1

    class NonEmptyClass:

        def __init__(self, a, b):
            if False:
                for i in range(10):
                    print('nop')
            self.a = a
            self.b = b
        inside = closured
    return NonEmptyClass(133, 135)

def simpleFunction19():
    if False:
        while True:
            i = 10
    lam = lambda l: l + 1
    return (lam(9), lam)

def simpleFunction20():
    if False:
        i = 10
        return i + 15
    try:
        a = []
        a[1]
    except IndexError as _e:
        pass

def simpleFunction21():
    if False:
        i = 10
        return i + 15

    class EmptyBaseClass:

        def base(self):
            if False:
                while True:
                    i = 10
            return 3

    class EmptyObjectClass(EmptyBaseClass):
        pass
    result = EmptyObjectClass()
    c = result.base()
    return (result, c)

def simpleFunction22():
    if False:
        return 10
    return True is False and False is not None

def simpleFunction23():
    if False:
        return 10
    not 2

def simpleFunction24p(x):
    if False:
        print('Hello World!')
    pass

def simpleFunction24():
    if False:
        i = 10
        return i + 15
    simpleFunction24p(x=3)

def simpleFunction25():
    if False:
        print('Hello World!')

    class X:
        f = 1

    def inplace_adder(b):
        if False:
            while True:
                i = 10
        X.f += b
    return inplace_adder(6 ** 8)

def simpleFunction26():
    if False:
        print('Hello World!')

    class X:
        f = [5]

    def inplace_adder(b):
        if False:
            return 10
        X.f += b
    return inplace_adder([1, 2])

def simpleFunction27():
    if False:
        i = 10
        return i + 15
    a = {'g': 8}

    def inplace_adder(b):
        if False:
            while True:
                i = 10
        a['g'] += b
    return inplace_adder(3)

def simpleFunction28():
    if False:
        for i in range(10):
            print('nop')
    a = {'g': [8], 'h': 2}

    def inplace_adder(b):
        if False:
            return 10
        a['g'] += b
    return inplace_adder([3, 5])

def simpleFunction29():
    if False:
        while True:
            i = 10
    return '3' in '7'

def simpleFunction30():
    if False:
        print('Hello World!')

    def generatorFunction():
        if False:
            i = 10
            return i + 15
        yield 1
        yield 2
        yield 3

def simpleFunction31():
    if False:
        while True:
            i = 10

    def generatorFunction():
        if False:
            return 10
        yield 1
        yield 2
        yield 3
    a = []
    for y in generatorFunction():
        a.append(y)
    for z in generatorFunction():
        a.append(z)

def simpleFunction32():
    if False:
        for i in range(10):
            print('nop')

    def generatorFunction():
        if False:
            while True:
                i = 10
        yield 1
    gen = generatorFunction()
    next(gen)

def simpleFunction33():
    if False:
        for i in range(10):
            print('nop')

    def generatorFunction():
        if False:
            print('Hello World!')
        a = 1
        yield a
    a = []
    for y in generatorFunction():
        a.append(y)

def simpleFunction34():
    if False:
        print('Hello World!')
    try:
        raise ValueError
    except:
        pass

def simpleFunction35():
    if False:
        while True:
            i = 10
    try:
        raise ValueError(1, 2, 3)
    except:
        pass

def simpleFunction36():
    if False:
        print('Hello World!')
    try:
        raise (TypeError, (3, x, x, x))
    except TypeError:
        pass

def simpleFunction37():
    if False:
        for i in range(10):
            print('nop')
    l = [1, 2, 3]
    try:
        (_a, _b) = l
    except ValueError:
        pass

def simpleFunction38():
    if False:
        while True:
            i = 10

    class Base:
        pass

    class Parent(Base):
        pass

def simpleFunction39():
    if False:
        while True:
            i = 10

    class Parent(object):
        pass

def simpleFunction40():
    if False:
        return 10

    def myGenerator():
        if False:
            print('Hello World!')
        yield 1
    myGenerator()

def simpleFunction41():
    if False:
        for i in range(10):
            print('nop')
    a = b = 2
    return (a, b)

def simpleFunction42():
    if False:
        i = 10
        return i + 15
    a = b = 2 * x
    return (a, b)

def simpleFunction43():
    if False:
        while True:
            i = 10

    class D:
        pass
    a = D()
    a.b = 1

def simpleFunction47():
    if False:
        print('Hello World!')

    def reraisy():
        if False:
            while True:
                i = 10

        def raisingFunction():
            if False:
                print('Hello World!')
            raise ValueError(3)

        def reraiser():
            if False:
                for i in range(10):
                    print('nop')
            raise
        try:
            raisingFunction()
        except:
            reraiser()
    try:
        reraisy()
    except:
        pass

def simpleFunction48():
    if False:
        print('Hello World!')

    class BlockExceptions:

        def __enter__(self):
            if False:
                print('Hello World!')
            pass

        def __exit__(self, exc, val, tb):
            if False:
                for i in range(10):
                    print('nop')
            return True
    with BlockExceptions():
        raise ValueError()
template = 'lala %s lala'

def simpleFunction49():
    if False:
        print('Hello World!')
    c = 3
    d = 4
    a = (x, y) = (b, e) = (c, d)
    return (a, y, b, e)
b = range(10)

def simpleFunction50():
    if False:
        i = 10
        return i + 15

    def getF():
        if False:
            while True:
                i = 10

        def f():
            if False:
                i = 10
                return i + 15
            for i in b:
                yield i
        return f
    f = getF()
    for x in range(2):
        _r = list(f())

def simpleFunction51():
    if False:
        for i in range(10):
            print('nop')
    g = (x for x in range(9))
    try:
        g.throw(ValueError, 9)
    except ValueError as _e:
        pass

def simpleFunction52():
    if False:
        return 10
    g = (x for x in range(9))
    try:
        g.throw(ValueError(9))
    except ValueError as _e:
        pass

def simpleFunction53():
    if False:
        print('Hello World!')
    g = (x for x in range(9))
    try:
        g.send(9)
    except TypeError as _e:
        pass

def simpleFunction54():
    if False:
        print('Hello World!')
    g = (x for x in range(9))
    next(g)
    try:
        g.send(9)
    except TypeError as _e:
        pass

def simpleFunction55():
    if False:
        return 10
    g = (x for x in range(9))
    try:
        g.close()
    except ValueError as _e:
        pass

def simpleFunction56():
    if False:
        for i in range(10):
            print('nop')
    'Throw into finished generator.'
    g = (x for x in range(9))
    list(g)
    try:
        g.throw(ValueError(9))
    except ValueError as _e:
        pass

def simpleFunction60():
    if False:
        while True:
            i = 10
    x = 1
    y = 2

    def f(a=x, b=y):
        if False:
            i = 10
            return i + 15
        return (a, b)
    f()
    f(2)
    f(3, 4)

def simpleFunction61():
    if False:
        while True:
            i = 10
    a = 3
    b = 5
    try:
        a = a * 2
        return a
    finally:
        a / b

def simpleFunction62():
    if False:
        return 10
    a = 3
    b = 5
    try:
        a = a * 2
        return a
    finally:
        return a / b

class X:

    def __del__(self):
        if False:
            return 10
        x = super()
        raise ValueError(1)

def simpleFunction63():
    if False:
        i = 10
        return i + 15

    def superUser():
        if False:
            while True:
                i = 10
        X()
    try:
        superUser()
    except Exception:
        pass

def simpleFunction64():
    if False:
        print('Hello World!')
    x = 2
    y = 3
    z = eval('x * y')
    return z

def simpleFunction65():
    if False:
        while True:
            i = 10
    import array
    a = array.array('b', b'')
    assert a == eval(repr(a), {'array': array.array})
    d = {'x': 2, 'y': 3}
    z = eval(repr(d), d)
    return z

def simpleFunction66():
    if False:
        return 10
    import types
    return type(simpleFunction65) == types.FunctionType

def simpleFunction67():
    if False:
        for i in range(10):
            print('nop')
    length = 100000
    pattern = '1234567890\x00\x01\x02\x03\x04\x05\x06'
    (q, r) = divmod(length, len(pattern))
    teststring = pattern * q + pattern[:r]
    return teststring

def simpleFunction68():
    if False:
        print('Hello World!')
    from random import randrange
    x = randrange(18)

def simpleFunction69():
    if False:
        print('Hello World!')
    pools = [tuple()]
    g = ((len(pool) == 0,) for pool in pools)
    next(g)

def simpleFunction70():
    if False:
        print('Hello World!')

    def gen():
        if False:
            for i in range(10):
                print('nop')
        try:
            undefined_yyy
        except Exception:
            pass
        yield sys.exc_info()
    try:
        undefined_xxx
    except Exception:
        return list(gen())

def simpleFunction71():
    if False:
        while True:
            i = 10
    try:
        undefined_global
    except Exception:
        try:
            try:
                raise
            finally:
                undefined_global
        except Exception:
            pass

def simpleFunction72():
    if False:
        print('Hello World!')
    try:
        for _i in range(10):
            try:
                undefined_global
            finally:
                break
    except Exception:
        pass

def simpleFunction73():
    if False:
        i = 10
        return i + 15
    for _i in range(10):
        try:
            undefined_global
        finally:
            return 7

def simpleFunction74():
    if False:
        return 10
    import os
    return os

def simpleFunction75():
    if False:
        for i in range(10):
            print('nop')

    def raising_gen():
        if False:
            for i in range(10):
                print('nop')
        try:
            raise TypeError
        except TypeError:
            yield
    g = raising_gen()
    next(g)
    try:
        g.throw(RuntimeError())
    except RuntimeError:
        pass

def simpleFunction76():
    if False:
        i = 10
        return i + 15

    class MyException(Exception):

        def __init__(self, obj):
            if False:
                for i in range(10):
                    print('nop')
            self.obj = obj

    class MyObj:
        pass

    def inner_raising_func():
        if False:
            return 10
        raise MyException(MyObj())
    try:
        inner_raising_func()
    except MyException:
        try:
            try:
                raise
            finally:
                raise
        except MyException:
            pass

class weirdstr(str):

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        return weirdstr(str.__getitem__(self, index))

def simpleFunction77():
    if False:
        i = 10
        return i + 15
    return filter(lambda x: x >= '33', weirdstr('1234'))

def simpleFunction78():
    if False:
        i = 10
        return i + 15
    a = 'x = 2'
    exec(a)

def simpleFunction79():
    if False:
        return 10
    'some doc'
    simpleFunction79.__doc__ = simpleFunction79.__doc__.replace('doc', 'dok')
    simpleFunction79.__doc__ += ' and more' + simpleFunction79.__name__

def simpleFunction80():
    if False:
        print('Hello World!')
    'some doc'
    del simpleFunction80.__doc__

def simpleFunction81():
    if False:
        return 10

    def f():
        if False:
            i = 10
            return i + 15
        yield 1
        j
    j = 1
    x = list(f())

def simpleFunction82():
    if False:
        while True:
            i = 10

    def f():
        if False:
            while True:
                i = 10
        yield 1
        j
    j = 1
    x = f.__doc__

def simpleFunction83():
    if False:
        i = 10
        return i + 15
    x = list(range(7))
    x[2] = 5
    j = 3
    x += [h * 2 for h in range(j)]

def simpleFunction84():
    if False:
        print('Hello World!')
    x = tuple(range(7))
    j = 3
    x += tuple([h * 2 for h in range(j)])

def simpleFunction85():
    if False:
        while True:
            i = 10
    x = list(range(7))
    x[2] = 3
    x *= 2

def simpleFunction86():
    if False:
        print('Hello World!')
    x = 'something'
    x += ''

def simpleFunction87():
    if False:
        while True:
            i = 10
    x = 7
    x += 2000

class C:

    def f(self):
        if False:
            print('Hello World!')
        pass

    def __iadd__(self, other):
        if False:
            return 10
        return self

    def method_function(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'x' in kwargs:
            x = kwargs['x']
            if type(x) is list:
                x.append(1)
        for x in args:
            if type(x) is list:
                x.append(1)
        return (args, kwargs)
    exec('\ndef method_uncompiled_function(*args, **kwargs):\n    # Make sure to mutate the list argument value\n    if "x" in kwargs:\n        x = kwargs["x"]\n\n        if type(x) is list:\n            x.append(1)\n\n    for x in args:\n        if type(x) is list:\n            x.append(1)\n\n\n    return args, kwargs\n')

    def method_function_with_defaults(self, a, b, c, d=1, e=2, f=3):
        if False:
            return 10
        return True

def simpleFunction88():
    if False:
        return 10
    x = C()
    x += C()

def simpleFunction89():
    if False:
        while True:
            i = 10
    x = [1, 2]
    x += [3, 4]

def anyArgs(*args, **kw):
    if False:
        return 10
    return (kw.keys(), kw.values())

def simpleFunction90():
    if False:
        return 10
    some_tuple = (simpleFunction89, simpleFunction89, simpleFunction89)
    anyArgs(*some_tuple)

def simpleFunction91():
    if False:
        print('Hello World!')
    some_dict = {'a': simpleFunction90}
    anyArgs(**some_dict)

def simpleFunction92():
    if False:
        for i in range(10):
            print('nop')
    some_tuple = (simpleFunction89,)
    some_dict = {'a': simpleFunction90}
    anyArgs(*some_tuple, **some_dict)

def simpleFunction93():
    if False:
        print('Hello World!')
    some_tuple = (simpleFunction89,)
    some_dict = {'a': simpleFunction90}
    anyArgs(some_tuple, *some_tuple, **some_dict)

def simpleFunction94():
    if False:
        return 10
    some_tuple = (simpleFunction89,)
    some_dict = {'a': simpleFunction90}
    anyArgs(*some_tuple, b=some_dict, **some_dict)

def simpleFunction95():
    if False:
        while True:
            i = 10
    some_tuple = (simpleFunction89,)
    some_dict = {'a': simpleFunction90}
    anyArgs(some_tuple, *some_tuple, b=some_dict, **some_dict)

def simpleFunction96():
    if False:
        while True:
            i = 10
    some_tuple = (simpleFunction89,)
    anyArgs(some_tuple, *some_tuple)

def simpleFunction97():
    if False:
        while True:
            i = 10
    some_dict = {'a': simpleFunction90, 'd': simpleFunction91}
    anyArgs(b=some_dict, c=1, **some_dict)

def simpleFunction98():
    if False:
        print('Hello World!')
    some_tuple = (simpleFunction89,)
    anyArgs(*some_tuple, b=some_tuple)

def simpleFunction99():
    if False:
        return 10
    some_dict = {'a': simpleFunction90}
    anyArgs(some_dict, **some_dict)

def simpleFunction100():
    if False:
        while True:
            i = 10

    def h(f):
        if False:
            i = 10
            return i + 15

        def g():
            if False:
                while True:
                    i = 10
            return f
        return g

    def f():
        if False:
            for i in range(10):
                print('nop')
        pass
    h(f)

def simpleFunction101():
    if False:
        i = 10
        return i + 15

    def orMaking(a, b):
        if False:
            print('Hello World!')
        x = 'axa'
        x += a or b
    orMaking('x', '')

class SomeClassWithAttributeAccess(object):
    READING = 1

    def use(self):
        if False:
            while True:
                i = 10
        return self.READING

def simpleFunction102():
    if False:
        print('Hello World!')
    SomeClassWithAttributeAccess().use()
    SomeClassWithAttributeAccess().use()

def getInt():
    if False:
        i = 10
        return i + 15
    return 3

def simpleFunction103():
    if False:
        return 10
    try:
        raise getInt()
    except TypeError:
        pass

class ClassWithGeneratorMethod:

    def generator_method(self):
        if False:
            print('Hello World!')
        yield self

def simpleFunction104():
    if False:
        return 10
    return list(ClassWithGeneratorMethod().generator_method())

def simpleFunction105():
    if False:
        i = 10
        return i + 15
    'Delete a started generator, not properly closing it before releasing.'

    def generator():
        if False:
            return 10
        yield 1
        yield 2
    g = generator()
    next(g)
    del g

def simpleFunction106():
    if False:
        i = 10
        return i + 15
    return sys.getsizeof(type)

def simpleFunction107():
    if False:
        for i in range(10):
            print('nop')
    return sum((i for i in range(x)))

def simpleFunction108():
    if False:
        return 10
    return sum((i for i in range(x)), 17)

def simpleFunction109():
    if False:
        for i in range(10):
            print('nop')
    sys.exc_info()

def simpleFunction110():
    if False:
        while True:
            i = 10

    def my_open(*args, **kwargs):
        if False:
            while True:
                i = 10
        return (args, kwargs)
    orig_open = __builtins__.open
    __builtins__.open = my_open
    open('me', buffering=True)
    __builtins__.open = orig_open
u = '__name__'

def simpleFunction111():
    if False:
        i = 10
        return i + 15
    return getattr(simpleFunction111, u)

def simpleFunction112():
    if False:
        for i in range(10):
            print('nop')
    TESTFN = 'tmp.txt'
    import codecs
    try:
        with open(TESTFN, 'wb') as out_file:
            out_file.write(b'\xa1')
        f = codecs.open(TESTFN, encoding='cp949')
        f.read(2)
    except UnicodeDecodeError:
        pass
    finally:
        try:
            f.close()
        except Exception:
            pass
        try:
            os.unlink(TESTFN)
        except Exception:
            pass

def simpleFunction113():
    if False:
        for i in range(10):
            print('nop')

    class A(object):
        pass
    a = A()
    a.a = a
    return a
l = []

def simpleFunction114():
    if False:
        for i in range(10):
            print('nop')
    global l
    l += ['something']
    del l[:]
i = 2 ** 16 + 1

def simpleFunction115():
    if False:
        for i in range(10):
            print('nop')
    global i
    i += 1
t = tuple(range(259))

def simpleFunction116():
    if False:
        return 10
    global t
    t += (2, 3)
    t = tuple(range(259))

def simpleFunction117():
    if False:
        print('Hello World!')
    try:
        return tuple(t) + i
    except TypeError:
        pass

def simpleFunction118():
    if False:
        for i in range(10):
            print('nop')
    try:
        return i + tuple(t)
    except TypeError:
        pass
t2 = tuple(range(9))

def simpleFunction119():
    if False:
        return 10
    return tuple(t) + t2

def simpleFunction120():
    if False:
        return 10
    return t2 + tuple(t)

def simpleFunction121():
    if False:
        return 10
    return tuple(t2) + tuple(t)

def simpleFunction122():
    if False:
        i = 10
        return i + 15
    try:
        return list(t) + i
    except TypeError:
        pass

def simpleFunction123():
    if False:
        return 10
    try:
        return i + list(t)
    except TypeError:
        pass
l2 = list(range(9))

def simpleFunction124():
    if False:
        while True:
            i = 10
    return list(t) + l2

def simpleFunction125():
    if False:
        while True:
            i = 10
    return l2 + list(t)

def simpleFunction126():
    if False:
        return 10
    return list(l2) + list(t)

class TupleWithSlots(tuple):

    def __add__(self, other):
        if False:
            return 10
        return 42

    def __radd__(self, other):
        if False:
            while True:
                i = 10
        return 42

def simpleFunction127():
    if False:
        i = 10
        return i + 15
    return tuple(t) + TupleWithSlots()

def simpleFunction128():
    if False:
        while True:
            i = 10
    return TupleWithSlots() + tuple(t)

class ListWithSlots(list):

    def __add__(self, other):
        if False:
            print('Hello World!')
        return 42

    def __radd__(self, other):
        if False:
            while True:
                i = 10
        return 42

def simpleFunction129():
    if False:
        while True:
            i = 10
    return list(t) + ListWithSlots()

def simpleFunction130():
    if False:
        print('Hello World!')
    return ListWithSlots() + list(t)

def simpleFunction131():
    if False:
        while True:
            i = 10
    try:
        C().f.__reduce__()
    except Exception as e:
        assert sys.version_info < (3, 4)

def simpleFunction132():
    if False:
        while True:
            i = 10
    C().f.__reduce_ex__(5)
x = 5

def local_function(*args, **kwargs):
    if False:
        while True:
            i = 10
    if 'x' in kwargs:
        x = kwargs['x']
        if type(x) is list:
            x.append(1)
    for x in args:
        if type(x) is list:
            x.append(1)
    return (args, kwargs)
exec('\ndef local_uncompiled_function(*args, **kwargs):\n    # Make sure to mutate the list argument value\n    if "x" in kwargs:\n        x = kwargs["x"]\n\n        if type(x) is list:\n            x.append(1)\n\n    for x in args:\n        if type(x) is list:\n            x.append(1)\n\n\n    return args, kwargs\n')

def simpleFunction133():
    if False:
        while True:
            i = 10
    local_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    local_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=1)
    local_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=x)
    local_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=[])
    local_function(1, 2, 3, 4, 5, 6, 7, x, 9, 10, 11, x=1)
    local_function(1, 2, 3, 4, 5, 6, 7, x, 9, 10, 11, x=x)
    local_function(1, 2, 3, 4, 5, 6, 7, [], 9, 10, 11, x=[])
    local_function(x=1)
    local_function(x=x)
    local_function(x=[])
    local_uncompiled_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    local_uncompiled_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=1)
    local_uncompiled_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=x)
    local_uncompiled_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=[])
    local_uncompiled_function(1, 2, 3, 4, 5, 6, 7, x, 9, 10, 11, x=1)
    local_uncompiled_function(1, 2, 3, 4, 5, 6, 7, x, 9, 10, 11, x=x)
    local_uncompiled_function(1, 2, 3, 4, 5, 6, 7, [], 9, 10, 11, x=[])
    local_uncompiled_function(x=1)
    local_uncompiled_function(x=x)
    local_uncompiled_function(x=[])
    c = C()
    C().method_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=1)
    C.method_function(c, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=1)
    C().method_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=x)
    C().method_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=[1])
    C.method_function(c, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=x)
    C().method_function(1, 2, 3, 4, 5, 6, 7, x, 9, 10, 11, x=1)
    C.method_function(c, 1, 2, 3, 4, 5, 6, 7, x, 9, 10, 11, x=1)
    C().method_function(1, 2, 3, 4, 5, 6, 7, x, 9, 10, 11, x=x)
    C.method_function(c, 1, 2, 3, 4, 5, 6, 7, x, 9, 10, 11, x=x)
    C().method_function(1, 2, 3, 4, 5, 6, 7, [1], 9, 10, 11, x=[1])
    C.method_function(c, 1, 2, 3, 4, 5, 6, 7, [1], 9, 10, 11, x=[1])
    C().method_function(x=1)
    C().method_function(x=x)
    C().method_function(x=[1])
    C().method_uncompiled_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=1)
    C.method_uncompiled_function(c, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=1)
    C().method_uncompiled_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=x)
    C().method_uncompiled_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=[1])
    C.method_uncompiled_function(c, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, x=x)
    C().method_uncompiled_function(1, 2, 3, 4, 5, 6, 7, x, 9, 10, 11, x=1)
    C.method_uncompiled_function(c, 1, 2, 3, 4, 5, 6, 7, x, 9, 10, 11, x=1)
    C().method_uncompiled_function(1, 2, 3, 4, 5, 6, 7, x, 9, 10, 11, x=x)
    C.method_uncompiled_function(c, 1, 2, 3, 4, 5, 6, 7, x, 9, 10, 11, x=x)
    C().method_uncompiled_function(1, 2, 3, 4, 5, 6, 7, [1], 9, 10, 11, x=[1])
    C.method_uncompiled_function(c, 1, 2, 3, 4, 5, 6, 7, [1], 9, 10, 11, x=[1])
    C().method_uncompiled_function(x=1)
    C().method_uncompiled_function(x=x)
    C().method_uncompiled_function(x=[1])
    C().method_function_with_defaults(1, 2, 3, d=1)
    C().method_function_with_defaults(1, x, 3, d=x)
    C().method_function_with_defaults(1, x, 3, d=[1])
tests_stderr = (63,)
tests_skipped = {}
result = executeReferenceChecked(prefix='simpleFunction', names=globals(), tests_skipped=tests_skipped, tests_stderr=tests_stderr)
sys.exit(0 if result else 1)