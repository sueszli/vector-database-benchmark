"""
Topic: 定义接口或抽象类
Desc : 
"""
from abc import ABCMeta, abstractmethod

class IStream(metaclass=ABCMeta):

    @abstractmethod
    def read(self, maxbytes=-1):
        if False:
            return 10
        pass

    @abstractmethod
    def write(self, data):
        if False:
            print('Hello World!')
        pass

class SocketStream(IStream):

    def read(self, maxbytes=-1):
        if False:
            for i in range(10):
                print('nop')
        pass

    def write(self, data):
        if False:
            while True:
                i = 10
        pass

class A(metaclass=ABCMeta):

    @property
    @abstractmethod
    def name(self):
        if False:
            i = 10
            return i + 15
        pass

    @name.setter
    @abstractmethod
    def name(self, value):
        if False:
            for i in range(10):
                print('nop')
        pass

    @classmethod
    @abstractmethod
    def method1(cls):
        if False:
            return 10
        pass

    @staticmethod
    @abstractmethod
    def method2():
        if False:
            for i in range(10):
                print('nop')
        pass