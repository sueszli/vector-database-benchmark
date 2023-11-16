"""
Topic: 利用函数注解实现方法重载
Desc : 
"""
import inspect
import types

class MultiMethod:
    """
    Represents a single multimethod.
    """

    def __init__(self, name):
        if False:
            print('Hello World!')
        self._methods = {}
        self.__name__ = name

    def register(self, meth):
        if False:
            i = 10
            return i + 15
        '\n        Register a new method as a multimethod\n        '
        sig = inspect.signature(meth)
        types = []
        for (name, parm) in sig.parameters.items():
            if name == 'self':
                continue
            if parm.annotation is inspect.Parameter.empty:
                raise TypeError('Argument {} must be annotated with a type'.format(name))
            if not isinstance(parm.annotation, type):
                raise TypeError('Argument {} annotation must be a type'.format(name))
            if parm.default is not inspect.Parameter.empty:
                self._methods[tuple(types)] = meth
            types.append(parm.annotation)
        self._methods[tuple(types)] = meth

    def __call__(self, *args):
        if False:
            while True:
                i = 10
        '\n        Call a method based on type signature of the arguments\n        '
        types = tuple((type(arg) for arg in args[1:]))
        meth = self._methods.get(types, None)
        if meth:
            return meth(*args)
        else:
            raise TypeError('No matching method for types {}'.format(types))

    def __get__(self, instance, cls):
        if False:
            while True:
                i = 10
        '\n        Descriptor method needed to make calls work in a class\n        '
        if instance is not None:
            return types.MethodType(self, instance)
        else:
            return self

class MultiDict(dict):
    """
    Special dictionary to build multimethods in a metaclass
    """

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        if key in self:
            current_value = self[key]
            if isinstance(current_value, MultiMethod):
                current_value.register(value)
            else:
                mvalue = MultiMethod(key)
                mvalue.register(current_value)
                mvalue.register(value)
                super().__setitem__(key, mvalue)
        else:
            super().__setitem__(key, value)

class MultipleMeta(type):
    """
    Metaclass that allows multiple dispatch of methods
    """

    def __new__(cls, clsname, bases, clsdict):
        if False:
            i = 10
            return i + 15
        return type.__new__(cls, clsname, bases, dict(clsdict))

    @classmethod
    def __prepare__(cls, clsname, bases):
        if False:
            for i in range(10):
                print('nop')
        return MultiDict()

class Spam(metaclass=MultipleMeta):

    def bar(self, x: int, y: int):
        if False:
            print('Hello World!')
        print('Bar 1:', x, y)

    def bar(self, s: str, n: int=0):
        if False:
            for i in range(10):
                print('nop')
        print('Bar 2:', s, n)
import time

class Date(metaclass=MultipleMeta):

    def __init__(self, year: int, month: int, day: int):
        if False:
            print('Hello World!')
        self.year = year
        self.month = month
        self.day = day

    def __init__(self):
        if False:
            while True:
                i = 10
        t = time.localtime()
        self.__init__(t.tm_year, t.tm_mon, t.tm_mday)
import types

class multimethod:

    def __init__(self, func):
        if False:
            for i in range(10):
                print('nop')
        self._methods = {}
        self.__name__ = func.__name__
        self._default = func

    def match(self, *types):
        if False:
            print('Hello World!')

        def register(func):
            if False:
                return 10
            ndefaults = len(func.__defaults__) if func.__defaults__ else 0
            for n in range(ndefaults + 1):
                self._methods[types[:len(types) - n]] = func
            return self
        return register

    def __call__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        types = tuple((type(arg) for arg in args[1:]))
        meth = self._methods.get(types, None)
        if meth:
            return meth(*args)
        else:
            return self._default(*args)

    def __get__(self, instance, cls):
        if False:
            while True:
                i = 10
        if instance is not None:
            return types.MethodType(self, instance)
        else:
            return self

class Spam:

    @multimethod
    def bar(self, *args):
        if False:
            print('Hello World!')
        raise TypeError('No matching method for bar')

    @bar.match(int, int)
    def bar(self, x, y):
        if False:
            return 10
        print('Bar 1:', x, y)

    @bar.match(str, int)
    def bar(self, s, n=0):
        if False:
            print('Hello World!')
        print('Bar 2:', s, n)