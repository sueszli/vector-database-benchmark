class CyTest(object):
    """
    >>> cy = CyTest()
    >>> '_CyTest__private' in dir(cy)
    True
    >>> cy._CyTest__private()
    8
    >>> '__private' in dir(cy)
    False
    >>> '_CyTest__x' in dir(cy)
    True

    >>> '__x' in dir(cy)
    False
    >>> cy._CyTest__y
    2

    >>> '_CyTest___more_than_two' in dir(cy)
    True
    >>> '___more_than_two' in dir(cy)
    False
    >>> '___more_than_two_special___' in dir(cy)
    True
    """
    __x = 1
    ___more_than_two = 3
    ___more_than_two_special___ = 4

    def __init__(self):
        if False:
            return 10
        self.__y = 2

    def __private(self):
        if False:
            print('Hello World!')
        return 8

    def get(self):
        if False:
            while True:
                i = 10
        '\n        >>> CyTest().get()\n        (1, 1, 8)\n        '
        return (self._CyTest__x, self.__x, self.__private())

    def get_inner(self):
        if False:
            return 10
        '\n        >>> CyTest().get_inner()\n        (1, 1, 8)\n        '

        def get(o):
            if False:
                for i in range(10):
                    print('nop')
            return (o._CyTest__x, o.__x, o.__private())
        return get(self)

class CyTestSub(CyTest):
    """
    >>> cy = CyTestSub()
    >>> '_CyTestSub__private' in dir(cy)
    True
    >>> cy._CyTestSub__private()
    9
    >>> '_CyTest__private' in dir(cy)
    True
    >>> cy._CyTest__private()
    8
    >>> '__private' in dir(cy)
    False

    >>> '_CyTestSub__x' in dir(cy)
    False
    >>> '_CyTestSub__y' in dir(cy)
    True
    >>> '_CyTest__x' in dir(cy)
    True
    >>> '__x' in dir(cy)
    False
    """
    __y = 2

    def __private(self):
        if False:
            return 10
        return 9

    def get(self):
        if False:
            i = 10
            return i + 15
        '\n        >>> CyTestSub().get()\n        (1, 2, 2, 9)\n        '
        return (self._CyTest__x, self._CyTestSub__y, self.__y, self.__private())

    def get_inner(self):
        if False:
            i = 10
            return i + 15
        '\n        >>> CyTestSub().get_inner()\n        (1, 2, 2, 9)\n        '

        def get(o):
            if False:
                for i in range(10):
                    print('nop')
            return (o._CyTest__x, o._CyTestSub__y, o.__y, o.__private())
        return get(self)

class _UnderscoreTest(object):
    """
    >>> ut = _UnderscoreTest()
    >>> '__x' in dir(ut)
    False
    >>> '_UnderscoreTest__x' in dir(ut)
    True
    >>> ut._UnderscoreTest__x
    1
    >>> ut.get()
    1
    >>> ut._UnderscoreTest__UnderscoreNested().ret1()
    1
    >>> ut._UnderscoreTest__UnderscoreNested.__name__
    '__UnderscoreNested'
    >>> ut._UnderscoreTest__prop
    1
    """
    __x = 1

    def get(self):
        if False:
            while True:
                i = 10
        return self.__x

    class __UnderscoreNested(object):

        def ret1(self):
            if False:
                i = 10
                return i + 15
            return 1

    @property
    def __prop(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__x

class C:
    error = 'Traceback (most recent call last):\n...\nTypeError:\n'
    __doc__ = "\n>>> instance = C()\n\nInstance methods have their arguments mangled\n>>> instance.method1(__arg=1) # doctest: +IGNORE_EXCEPTION_DETAIL\n{error}\n>>> instance.method1(_C__arg=1)\n1\n>>> instance.method2(__arg=1) # doctest: +IGNORE_EXCEPTION_DETAIL\n{error}\n>>> instance.method2(_C__arg=1)\n1\n\nWorks when optional argument isn't passed\n>>> instance.method2()\nNone\n\nWhere args are in the function's **kwargs dict, names aren't mangled\n>>> instance.method3(__arg=1) # doctest:\n1\n>>> instance.method3(_C__arg=1) # doctest: +IGNORE_EXCEPTION_DETAIL\nTraceback (most recent call last):\n...\nKeyError:\n\nLambda functions behave in the same way:\n>>> instance.method_lambda(__arg=1) # doctest: +IGNORE_EXCEPTION_DETAIL\n{error}\n>>> instance.method_lambda(_C__arg=1)\n1\n\nClass methods - have their arguments mangled\n>>> instance.class_meth(__arg=1) # doctest: +IGNORE_EXCEPTION_DETAIL\n{error}\n>>> instance.class_meth(_C__arg=1)\n1\n>>> C.class_meth(__arg=1) # doctest: +IGNORE_EXCEPTION_DETAIL\n{error}\n>>> C.class_meth(_C__arg=1)\n1\n\nStatic methods - have their arguments mangled\n>>> instance.static_meth(__arg=1) # doctest: +IGNORE_EXCEPTION_DETAIL\n{error}\n>>> instance.static_meth(_C__arg=1)\n1\n>>> C.static_meth(__arg=1) # doctest: +IGNORE_EXCEPTION_DETAIL\n{error}\n>>> C.static_meth(_C__arg=1)\n1\n\nFunctions assigned to the class don't have their arguments mangled\n>>> instance.class_assigned_function(__arg=1)\n1\n>>> instance.class_assigned_function(_C__arg=1) # doctest: +IGNORE_EXCEPTION_DETAIL\n{error}\n\nFunctions assigned to an instance don't have their arguments mangled\n>>> instance.instance_assigned_function = free_function2\n>>> instance.instance_assigned_function(__arg=1)\n1\n>>> instance.instance_assigned_function(_C__arg=1) # doctest: +IGNORE_EXCEPTION_DETAIL\n{error}\n\nLocals are reported as mangled\n>>> list(sorted(k for k in instance.get_locals(1).keys()))\n['_C__arg', 'self']\n".format(error=error)

    def method1(self, __arg):
        if False:
            while True:
                i = 10
        print(__arg)

    def method2(self, __arg=None):
        if False:
            while True:
                i = 10
        print(__arg)

    def method3(self, **kwargs):
        if False:
            print('Hello World!')
        print(kwargs['__arg'])
    method_lambda = lambda self, __arg: __arg

    def get_locals(self, __arg):
        if False:
            print('Hello World!')
        return locals()

    @classmethod
    def class_meth(cls, __arg):
        if False:
            for i in range(10):
                print('nop')
        print(__arg)

    @staticmethod
    def static_meth(__arg, dummy_arg=None):
        if False:
            while True:
                i = 10
        print(__arg)

def free_function1(x, __arg):
    if False:
        for i in range(10):
            print('nop')
    print(__arg)

def free_function2(__arg, dummy_arg=None):
    if False:
        for i in range(10):
            print('nop')
    print(__arg)
C.class_assigned_function = free_function1
__global_arg = True
_D__arg1 = None
_D__global_arg = False

def can_find_global_arg():
    if False:
        while True:
            i = 10
    '\n    >>> can_find_global_arg()\n    True\n    '
    return __global_arg

def cant_find_global_arg():
    if False:
        while True:
            i = 10
    '\n    Gets _D_global_arg instead\n    >>> cant_find_global_arg()\n    False\n    '

    class D:

        def f(self):
            if False:
                for i in range(10):
                    print('nop')
            return __global_arg
    return D().f()

class CMultiplyNested:

    def f1(self, __arg, name=None, return_closure=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        >>> inst = CMultiplyNested()\n        >>> for name in [None, \'__arg\', \'_CMultiplyNested__arg\', \'_D__arg\']:\n        ...    try:\n        ...        print(inst.f1(1,name))\n        ...    except TypeError:\n        ...        print("TypeError") # not concerned about exact details\n        ...    # now test behaviour is the same in closures\n        ...    closure = inst.f1(1, return_closure=True)\n        ...    try:\n        ...        if name is None:\n        ...            print(closure(2))\n        ...        else:\n        ...            print(closure(**{ name: 2}))\n        ...    except TypeError:\n        ...        print("TypeError")\n        2\n        2\n        TypeError\n        TypeError\n        TypeError\n        TypeError\n        2\n        2\n        '

        class D:

            def g(self, __arg):
                if False:
                    for i in range(10):
                        print('nop')
                return __arg
        if return_closure:
            return D().g
        if name is not None:
            return D().g(**{name: 2})
        else:
            return D().g(2)

    def f2(self, __arg1):
        if False:
            return 10
        "\n        This finds the global name '_D__arg1'\n        It's tested in this way because without the global\n        Python gives a runtime error and Cython a compile error\n        >>> print(CMultiplyNested().f2(1))\n        None\n        "

        class D:

            def g(self):
                if False:
                    for i in range(10):
                        print('nop')
                return __arg1
        return D().g()

    def f3(self, arg, name):
        if False:
            i = 10
            return i + 15
        "\n        >>> inst = CMultiplyNested()\n        >>> inst.f3(1, None)\n        2\n        >>> inst.f3(1, '__arg') # doctest: +IGNORE_EXCEPTION_DETAIL\n        Traceback (most recent call last):\n        ...\n        TypeError:\n        >>> inst.f3(1, '_CMultiplyNested__arg')\n        2\n        "

        def g(__arg, dummy=1):
            if False:
                while True:
                    i = 10
            return __arg
        if name is not None:
            return g(**{name: 2})
        else:
            return g(2)

    def f4(self, __arg):
        if False:
            print('Hello World!')
        '\n        >>> CMultiplyNested().f4(1)\n        1\n        '

        def g():
            if False:
                return 10
            return __arg
        return g()

    def f5(self, __arg):
        if False:
            return 10
        '\n        Default values are found in the outer scope correctly\n        >>> CMultiplyNested().f5(1)\n        1\n        '

        def g(x=__arg):
            if False:
                for i in range(10):
                    print('nop')
            return x
        return g()

    def f6(self, __arg1):
        if False:
            for i in range(10):
                print('nop')
        '\n        This will find the global name _D__arg1\n        >>> print(CMultiplyNested().f6(1))\n        None\n        '

        class D:

            def g(self, x=__arg1):
                if False:
                    for i in range(10):
                        print('nop')
                return x
        return D().g()

    def f7(self, __arg):
        if False:
            return 10
        '\n        Lookup works in generator expressions\n        >>> list(CMultiplyNested().f7(1))\n        [1]\n        '
        return (__arg for x in range(1))

class __NameWithDunder:
    """
    >>> __NameWithDunder.__name__
    '__NameWithDunder'
    """
    pass

class Inherits(__NameWithDunder):
    """
    Compile check that it can find the base class
    >>> x = Inherits()
    """
    pass

def regular_function(__x, dummy=None):
    if False:
        for i in range(10):
            print('nop')
    return __x

class CallsRegularFunction:

    def call(self):
        if False:
            while True:
                i = 10
        '\n        >>> CallsRegularFunction().call()\n        1\n        '
        return regular_function(__x=1)