"""
Topic: sample
Desc : 
"""
import time
from functools import wraps

def timethis(func):
    if False:
        while True:
            i = 10

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        print(end - start)
        return r
    return wrapper

class Spam:

    @timethis
    def instance_method(self, n):
        if False:
            i = 10
            return i + 15
        print(self, n)
        while n > 0:
            n -= 1

    @classmethod
    @timethis
    def class_method(cls, n):
        if False:
            i = 10
            return i + 15
        print(cls, n)
        while n > 0:
            n -= 1

    @staticmethod
    @timethis
    def static_method(n):
        if False:
            i = 10
            return i + 15
        print(n)
        while n > 0:
            n -= 1
from abc import ABCMeta, abstractmethod

class A(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def method(cls):
        if False:
            return 10
        pass