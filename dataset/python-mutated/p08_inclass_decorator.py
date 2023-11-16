"""
Topic: 类中定义装饰器
Desc : 
"""
from functools import wraps

class A:

    def decorator1(self, func):
        if False:
            i = 10
            return i + 15

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            print('Decorator 1')
            return func(*args, **kwargs)
        return wrapper

    @classmethod
    def decorator2(cls, func):
        if False:
            i = 10
            return i + 15

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                return 10
            print('Decorator 2')
            return func(*args, **kwargs)
        return wrapper

class Person:
    first_name = property()

    @first_name.getter
    def first_name(self):
        if False:
            while True:
                i = 10
        return self._first_name

    @first_name.setter
    def first_name(self, value):
        if False:
            while True:
                i = 10
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value