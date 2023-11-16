"""
Topic: 元类中的可选参数
Desc : 
"""
from abc import ABCMeta, abstractmethod

class IStream(metaclass=ABCMeta):

    @abstractmethod
    def read(self, maxsize=None):
        if False:
            return 10
        pass

    @abstractmethod
    def write(self, data):
        if False:
            return 10
        pass

class MyMeta(type):

    @classmethod
    def __prepare__(cls, name, bases, *, debug=False, synchronize=False):
        if False:
            while True:
                i = 10
        pass
        return super().__prepare__(name, bases)

    def __new__(cls, name, bases, ns, *, debug=False, synchronize=False):
        if False:
            for i in range(10):
                print('nop')
        pass
        return super().__new__(cls, name, bases, ns)

    def __init__(self, name, bases, ns, *, debug=False, synchronize=False):
        if False:
            for i in range(10):
                print('nop')
        pass
        super().__init__(name, bases, ns)