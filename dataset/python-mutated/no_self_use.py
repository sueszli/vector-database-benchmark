import abc
from typing_extensions import override

class Person:

    def developer_greeting(self, name):
        if False:
            return 10
        print(f'Greetings {name}!')

    def greeting_1(self):
        if False:
            for i in range(10):
                print('nop')
        print('Hello!')

    def greeting_2(self):
        if False:
            i = 10
            return i + 15
        print('Hi!')

def developer_greeting():
    if False:
        while True:
            i = 10
    print('Greetings developer!')

class Person:
    name = 'Paris'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def __cmp__(self, other):
        if False:
            while True:
                i = 10
        print(24)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'Person'

    def func(self):
        if False:
            return 10
        ...

    def greeting_1(self):
        if False:
            i = 10
            return i + 15
        print(f'Hello from {self.name} !')

    @staticmethod
    def greeting_2():
        if False:
            while True:
                i = 10
        print('Hi!')

class Base(abc.ABC):
    """abstract class"""

    @abstractmethod
    def abstract_method(self):
        if False:
            return 10
        'abstract method could not be a function'
        raise NotImplementedError

class Sub(Base):

    @override
    def abstract_method(self):
        if False:
            print('Hello World!')
        print('concrete method')

class Prop:

    @property
    def count(self):
        if False:
            while True:
                i = 10
        return 24

class A:

    def foo(self):
        if False:
            print('Hello World!')
        ...

class B(A):

    @override
    def foo(self):
        if False:
            print('Hello World!')
        ...

    def foobar(self):
        if False:
            while True:
                i = 10
        super()

    def bar(self):
        if False:
            i = 10
            return i + 15
        super().foo()

    def baz(self):
        if False:
            for i in range(10):
                print('nop')
        if super().foo():
            ...