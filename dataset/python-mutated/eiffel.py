"""
Support Eiffel-style preconditions and postconditions for functions.

An example for Python metaclasses.
"""
import unittest
from types import FunctionType as function

class EiffelBaseMetaClass(type):

    def __new__(meta, name, bases, dict):
        if False:
            while True:
                i = 10
        meta.convert_methods(dict)
        return super(EiffelBaseMetaClass, meta).__new__(meta, name, bases, dict)

    @classmethod
    def convert_methods(cls, dict):
        if False:
            for i in range(10):
                print('nop')
        'Replace functions in dict with EiffelMethod wrappers.\n\n        The dict is modified in place.\n\n        If a method ends in _pre or _post, it is removed from the dict\n        regardless of whether there is a corresponding method.\n        '
        methods = []
        for (k, v) in dict.items():
            if k.endswith('_pre') or k.endswith('_post'):
                assert isinstance(v, function)
            elif isinstance(v, function):
                methods.append(k)
        for m in methods:
            pre = dict.get('%s_pre' % m)
            post = dict.get('%s_post' % m)
            if pre or post:
                dict[m] = cls.make_eiffel_method(dict[m], pre, post)

class EiffelMetaClass1(EiffelBaseMetaClass):

    @staticmethod
    def make_eiffel_method(func, pre, post):
        if False:
            for i in range(10):
                print('nop')

        def method(self, *args, **kwargs):
            if False:
                print('Hello World!')
            if pre:
                pre(self, *args, **kwargs)
            rv = func(self, *args, **kwargs)
            if post:
                post(self, rv, *args, **kwargs)
            return rv
        if func.__doc__:
            method.__doc__ = func.__doc__
        return method

class EiffelMethodWrapper:

    def __init__(self, inst, descr):
        if False:
            for i in range(10):
                print('nop')
        self._inst = inst
        self._descr = descr

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self._descr.callmethod(self._inst, args, kwargs)

class EiffelDescriptor:

    def __init__(self, func, pre, post):
        if False:
            return 10
        self._func = func
        self._pre = pre
        self._post = post
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, obj, cls=None):
        if False:
            i = 10
            return i + 15
        return EiffelMethodWrapper(obj, self)

    def callmethod(self, inst, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        if self._pre:
            self._pre(inst, *args, **kwargs)
        x = self._func(inst, *args, **kwargs)
        if self._post:
            self._post(inst, x, *args, **kwargs)
        return x

class EiffelMetaClass2(EiffelBaseMetaClass):
    make_eiffel_method = EiffelDescriptor

class Tests(unittest.TestCase):

    def testEiffelMetaClass1(self):
        if False:
            return 10
        self._test(EiffelMetaClass1)

    def testEiffelMetaClass2(self):
        if False:
            while True:
                i = 10
        self._test(EiffelMetaClass2)

    def _test(self, metaclass):
        if False:
            print('Hello World!')

        class Eiffel(metaclass=metaclass):
            pass

        class Test(Eiffel):

            def m(self, arg):
                if False:
                    print('Hello World!')
                'Make it a little larger'
                return arg + 1

            def m2(self, arg):
                if False:
                    print('Hello World!')
                'Make it a little larger'
                return arg + 1

            def m2_pre(self, arg):
                if False:
                    while True:
                        i = 10
                assert arg > 0

            def m2_post(self, result, arg):
                if False:
                    for i in range(10):
                        print('nop')
                assert result > arg

        class Sub(Test):

            def m2(self, arg):
                if False:
                    return 10
                return arg ** 2

            def m2_post(self, Result, arg):
                if False:
                    i = 10
                    return i + 15
                super(Sub, self).m2_post(Result, arg)
                assert Result < 100
        t = Test()
        self.assertEqual(t.m(1), 2)
        self.assertEqual(t.m2(1), 2)
        self.assertRaises(AssertionError, t.m2, 0)
        s = Sub()
        self.assertRaises(AssertionError, s.m2, 1)
        self.assertRaises(AssertionError, s.m2, 10)
        self.assertEqual(s.m2(5), 25)
if __name__ == '__main__':
    unittest.main()