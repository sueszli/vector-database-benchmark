"""
python_decorator.py by xianhu
"""
import functools

def logging(func):
    if False:
        return 10

    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if False:
            return 10
        print('%s called' % func.__name__)
        result = func(*args, **kwargs)
        print('%s end' % func.__name__)
        return result
    return decorator

@logging
def test01(a, b):
    if False:
        for i in range(10):
            print('nop')
    print('in function test01, a=%s, b=%s' % (a, b))
    return 1

@logging
def test02(a, b, c=1):
    if False:
        for i in range(10):
            print('nop')
    print('in function test02, a=%s, b=%s, c=%s' % (a, b, c))
    return 1

def params_chack(*types, **kwtypes):
    if False:
        print('Hello World!')

    def _outer(func):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(func)
        def _inner(*args, **kwargs):
            if False:
                while True:
                    i = 10
            result = [isinstance(_param, _type) for (_param, _type) in zip(args, types)]
            assert all(result), 'params_chack: invalid parameters'
            result = [isinstance(kwargs[_param], kwtypes[_param]) for _param in kwargs if _param in kwtypes]
            assert all(result), 'params_chack: invalid parameters'
            return func(*args, **kwargs)
        return _inner
    return _outer

@params_chack(int, (list, tuple))
def test03(a, b):
    if False:
        print('Hello World!')
    print('in function test03, a=%s, b=%s' % (a, b))
    return 1

@params_chack(int, str, c=(int, str))
def test04(a, b, c):
    if False:
        return 10
    print('in function test04, a=%s, b=%s, c=%s' % (a, b, c))
    return 1

class ATest(object):

    @params_chack(object, int, str)
    def test(self, a, b):
        if False:
            while True:
                i = 10
        print('in function test of ATest, a=%s, b=%s' % (a, b))
        return 1

@logging
@params_chack(int, str, (list, tuple))
def test05(a, b, c):
    if False:
        i = 10
        return i + 15
    print('in function test05, a=%s, b=%s, c=%s' % (a, b, c))
    return 1

class Decorator(object):

    def __init__(self, func):
        if False:
            while True:
                i = 10
        self.func = func
        return

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        print('%s called' % self.func.__name__)
        result = self.func(*args, **kwargs)
        print('%s end' % self.func.__name__)
        return result

@Decorator
def test06(a, b, c):
    if False:
        for i in range(10):
            print('nop')
    print('in function test06, a=%s, b=%s, c=%s' % (a, b, c))
    return 1

class ParamCheck(object):

    def __init__(self, *types, **kwtypes):
        if False:
            return 10
        self.types = types
        self.kwtypes = kwtypes
        return

    def __call__(self, func):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(func)
        def _inner(*args, **kwargs):
            if False:
                while True:
                    i = 10
            result = [isinstance(_param, _type) for (_param, _type) in zip(args, self.types)]
            assert all(result), 'params_chack: invalid parameters'
            result = [isinstance(kwargs[_param], self.kwtypes[_param]) for _param in kwargs if _param in self.kwtypes]
            assert all(result), 'params_chack: invalid parameters'
            return func(*args, **kwargs)
        return _inner

@ParamCheck(int, str, (list, tuple))
def test07(a, b, c):
    if False:
        while True:
            i = 10
    print('in function test06, a=%s, b=%s, c=%s' % (a, b, c))
    return 1

def funccache(func):
    if False:
        while True:
            i = 10
    cache = {}

    @functools.wraps(func)
    def _inner(*args):
        if False:
            for i in range(10):
                print('nop')
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return _inner

@funccache
def test08(a, b, c):
    if False:
        i = 10
        return i + 15
    return a + b + c

class Person(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._name = None
        return

    def get_name(self):
        if False:
            i = 10
            return i + 15
        print('get_name')
        return self._name

    def set_name(self, name):
        if False:
            while True:
                i = 10
        print('set_name')
        self._name = name
        return
    name = property(fget=get_name, fset=set_name, doc='person name')

class People(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._name = None
        self._age = None
        return

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return self._name

    @name.setter
    def name(self, name):
        if False:
            for i in range(10):
                print('nop')
        self._name = name
        return

    @property
    def age(self):
        if False:
            i = 10
            return i + 15
        return self._age

    @age.setter
    def age(self, age):
        if False:
            for i in range(10):
                print('nop')
        assert 0 < age < 120
        self._age = age
        return

class A(object):
    var = 1

    def func(self):
        if False:
            return 10
        print(self.var)
        return

    @staticmethod
    def static_func():
        if False:
            for i in range(10):
                print('nop')
        print(A.var)
        return

    @classmethod
    def class_func(cls):
        if False:
            i = 10
            return i + 15
        print(cls.var)
        cls().func()
        return