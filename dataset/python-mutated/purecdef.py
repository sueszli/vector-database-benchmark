import cython
from cython import cfunc, cclass, ccall

@cython.test_assert_path_exists('//CFuncDefNode')
@cython.cfunc
def ftang():
    if False:
        i = 10
        return i + 15
    x = 0

@cython.test_assert_path_exists('//CFuncDefNode')
@cfunc
def fpure(a):
    if False:
        while True:
            i = 10
    return a * 2

def test():
    if False:
        return 10
    '\n    >>> test()\n    4\n    '
    ftang()
    return fpure(2)
with cfunc:

    @cython.test_assert_path_exists('//CFuncDefNode')
    def fwith1(a):
        if False:
            print('Hello World!')
        return a * 3

    @cython.test_assert_path_exists('//CFuncDefNode')
    def fwith2(a):
        if False:
            print('Hello World!')
        return a * 4

    @cython.test_assert_path_exists('//CFuncDefNode', '//LambdaNode', '//GeneratorDefNode', '//GeneratorBodyDefNode')
    def f_with_genexpr(a):
        if False:
            while True:
                i = 10
        f = lambda x: x + 1
        return (f(x) for x in a)
with cclass:

    @cython.test_assert_path_exists('//CClassDefNode')
    class Egg(object):
        pass

    @cython.test_assert_path_exists('//CClassDefNode')
    class BigEgg(object):

        @cython.test_assert_path_exists('//CFuncDefNode')
        @cython.cfunc
        def f(self, a):
            if False:
                while True:
                    i = 10
            return a * 10

def test_with():
    if False:
        return 10
    '\n    >>> test_with()\n    (3, 4, 50)\n    '
    return (fwith1(1), fwith2(1), BigEgg().f(5))

@cython.test_assert_path_exists('//CClassDefNode')
@cython.cclass
class PureFoo(object):
    a = cython.declare(cython.double)

    def __init__(self, a):
        if False:
            i = 10
            return i + 15
        self.a = a

    def __call__(self):
        if False:
            return 10
        return self.a

    @cython.test_assert_path_exists('//CFuncDefNode')
    @cython.cfunc
    def puremeth(self, a):
        if False:
            i = 10
            return i + 15
        return a * 2

def test_method():
    if False:
        print('Hello World!')
    '\n    >>> test_method()\n    4\n    True\n    '
    x = PureFoo(2)
    print(x.puremeth(2))
    if cython.compiled:
        print(isinstance(x(), float))
    else:
        print(True)
    return

@cython.ccall
def ccall_sqr(x):
    if False:
        return 10
    return x * x

@cclass
class Overridable(object):

    @ccall
    def meth(self):
        if False:
            return 10
        return 0

def test_ccall():
    if False:
        while True:
            i = 10
    '\n    >>> test_ccall()\n    25\n    >>> ccall_sqr(5)\n    25\n    '
    return ccall_sqr(5)

def test_ccall_method(x):
    if False:
        i = 10
        return i + 15
    '\n    >>> test_ccall_method(Overridable())\n    0\n    >>> Overridable().meth()\n    0\n    >>> class Foo(Overridable):\n    ...    def meth(self):\n    ...        return 1\n    >>> test_ccall_method(Foo())\n    1\n    >>> Foo().meth()\n    1\n    '
    return x.meth()

@cython.cfunc
@cython.returns(cython.p_int)
@cython.locals(xptr=cython.p_int)
def typed_return(xptr):
    if False:
        i = 10
        return i + 15
    return xptr

def test_typed_return():
    if False:
        i = 10
        return i + 15
    '\n    >>> test_typed_return()\n    '
    x = cython.declare(int, 5)
    assert typed_return(cython.address(x))[0] is x

def test_genexpr_in_cdef(l):
    if False:
        print('Hello World!')
    '\n    >>> gen = test_genexpr_in_cdef([1, 2, 3])\n    >>> list(gen)\n    [2, 3, 4]\n    >>> list(gen)\n    []\n    '
    return f_with_genexpr(l)