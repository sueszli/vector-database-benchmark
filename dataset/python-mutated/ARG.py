from abc import abstractmethod
from typing import overload, cast
from typing_extensions import override

def f(self, x):
    if False:
        for i in range(10):
            print('nop')
    print('Hello, world!')

def f(cls, x):
    if False:
        return 10
    print('Hello, world!')

def f(self, x):
    if False:
        for i in range(10):
            print('nop')
    ...

def f(cls, x):
    if False:
        return 10
    ...
lambda x: print('Hello, world!')
lambda : print('Hello, world!')

class C:

    def f(self, x):
        if False:
            while True:
                i = 10
        print('Hello, world!')

    def f(self, /, x):
        if False:
            for i in range(10):
                print('nop')
        print('Hello, world!')

    def f(cls, x):
        if False:
            while True:
                i = 10
        print('Hello, world!')

    @classmethod
    def f(cls, x):
        if False:
            while True:
                i = 10
        print('Hello, world!')

    @staticmethod
    def f(cls, x):
        if False:
            i = 10
            return i + 15
        print('Hello, world!')

    @staticmethod
    def f(x):
        if False:
            return 10
        print('Hello, world!')

    def f(self, x):
        if False:
            return 10
        ...

    def f(self, /, x):
        if False:
            return 10
        ...

    def f(cls, x):
        if False:
            for i in range(10):
                print('nop')
        ...

    @classmethod
    def f(cls, x):
        if False:
            i = 10
            return i + 15
        ...

    @staticmethod
    def f(cls, x):
        if False:
            print('Hello World!')
        ...

    @staticmethod
    def f(x):
        if False:
            i = 10
            return i + 15
        ...

    def f(self, x):
        if False:
            return 10
        'Docstring.'

    def f(self, x):
        if False:
            for i in range(10):
                print('nop')
        'Docstring.'
        ...

    def f(self, x):
        if False:
            for i in range(10):
                print('nop')
        pass

    def f(self, x):
        if False:
            return 10
        raise NotImplementedError

    def f(self, x):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def f(self, x):
        if False:
            while True:
                i = 10
        raise NotImplementedError('...')

    def f(self, x):
        if False:
            i = 10
            return i + 15
        raise NotImplemented

    def f(self, x):
        if False:
            i = 10
            return i + 15
        raise NotImplemented()

    def f(self, x):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplemented('...')

    @abstractmethod
    def f(self, x):
        if False:
            i = 10
            return i + 15
        print('Hello, world!')

    @abstractmethod
    def f(self, /, x):
        if False:
            while True:
                i = 10
        print('Hello, world!')

    @abstractmethod
    def f(cls, x):
        if False:
            i = 10
            return i + 15
        print('Hello, world!')

    @classmethod
    @abstractmethod
    def f(cls, x):
        if False:
            for i in range(10):
                print('nop')
        print('Hello, world!')

    @staticmethod
    @abstractmethod
    def f(cls, x):
        if False:
            i = 10
            return i + 15
        print('Hello, world!')

    @staticmethod
    @abstractmethod
    def f(x):
        if False:
            i = 10
            return i + 15
        print('Hello, world!')

    @override
    def f(self, x):
        if False:
            i = 10
            return i + 15
        print('Hello, world!')

    @override
    def f(self, /, x):
        if False:
            while True:
                i = 10
        print('Hello, world!')

    @override
    def f(cls, x):
        if False:
            print('Hello World!')
        print('Hello, world!')

    @classmethod
    @override
    def f(cls, x):
        if False:
            i = 10
            return i + 15
        print('Hello, world!')

    @staticmethod
    @override
    def f(cls, x):
        if False:
            print('Hello World!')
        print('Hello, world!')

    @staticmethod
    @override
    def f(x):
        if False:
            while True:
                i = 10
        print('Hello, world!')

@overload
def f(a: str, b: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def f(a: int, b: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    ...

def f(a, b):
    if False:
        return 10
    return f'{a}{b}'

class C:

    def __init__(self, x) -> None:
        if False:
            while True:
                i = 10
        print('Hello, world!')

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Hello, world!'

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if False:
            return 10
        print('Hello, world!')

def f(x: None) -> None:
    if False:
        for i in range(10):
            print('nop')
    _ = cast(Any, _identity)(x=x)

def f(bar: str):
    if False:
        for i in range(10):
            print('nop')
    print(locals())

class C:

    def __init__(self, x) -> None:
        if False:
            return 10
        print(locals())