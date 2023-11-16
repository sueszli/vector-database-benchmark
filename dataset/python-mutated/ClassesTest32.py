""" Covers call order of Python3 meta classes. """
try:
    from collections.abc import OrderedDict
except ImportError:
    from collections import OrderedDict
print('Call order of Python3 meta classes:')

def a():
    if False:
        i = 10
        return i + 15
    x = 1

    class A:
        print('Class body a.A is evaluating closure x', x)
    print('Called', a)
    return A

def b():
    if False:
        return 10

    class B:
        pass
    print('Called', b)
    return B

def displayable(dictionary):
    if False:
        i = 10
        return i + 15
    return sorted(dictionary.items())

def m():
    if False:
        print('Hello World!')

    class M(type):

        def __new__(cls, class_name, bases, attrs, **over):
            if False:
                i = 10
                return i + 15
            print('Metaclass M.__new__ cls', cls, 'name', class_name, 'bases', bases, 'dict', displayable(attrs), 'extra class defs', displayable(over))
            return type.__new__(cls, class_name, bases, attrs)

        def __init__(self, name, bases, attrs, **over):
            if False:
                for i in range(10):
                    print('nop')
            print('Metaclass M.__init__', name, bases, displayable(attrs), displayable(over))
            super().__init__(name, bases, attrs)

        def __prepare__(cls, bases, **over):
            if False:
                print('Hello World!')
            print('Metaclass M.__prepare__', cls, bases, displayable(over))
            return OrderedDict()
    print('Called', m)
    return M

def d():
    if False:
        for i in range(10):
            print('nop')
    print('Called', d)
    return 1

def e():
    if False:
        while True:
            i = 10
    print('Called', e)
    return 2

class C1(a(), b(), other=d(), metaclass=m(), yet_other=e()):
    import sys
print('OK, class created', C1)
print('Attribute C1.__dict__ has type', type(C1.__dict__))
print('Function local classes can be made global and get proper __qualname__:')

def someFunctionWithLocalClassesMadeGlobal():
    if False:
        print('Hello World!')
    global C

    class C:
        pass

        class D:
            pass
        try:
            print('Nested class qualname is', D.__qualname__)
        except AttributeError:
            pass
    try:
        print('Local class made global qualname is', C.__qualname__)
    except AttributeError:
        pass
someFunctionWithLocalClassesMadeGlobal()
print('Function in a class with private name')

class someClassWithPrivateArgumentNames:

    def f(self, *, __kw: 1):
        if False:
            while True:
                i = 10
        pass
print(someClassWithPrivateArgumentNames.f.__annotations__)
print('OK.')