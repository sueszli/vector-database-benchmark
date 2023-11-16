from builtins import _test_source, _test_sink
from typing import TypeVar
from typing_extensions import Self
TFoo = TypeVar('TFoo', bound='Foo')

class Foo:
    tainted_class: str = ''
    not_tainted_class: str = ''

    def __init__(self, tainted_instance: str, not_tainted_instance: str) -> None:
        if False:
            i = 10
            return i + 15
        self.tainted_instance: str = tainted_instance
        self.not_tainted_instance: str = not_tainted_instance
        self.tainted_extra_instance: str = _test_source()
        self.not_tainted_extra_instance: str = ''

    def untyped_self_class_direct(self) -> None:
        if False:
            return 10
        _test_sink(self.__class__.tainted_class)

    def untyped_self_class(self) -> None:
        if False:
            print('Hello World!')
        _test_sink(self.tainted_class)

    def untyped_self_instance(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        _test_sink(self.tainted_instance)

    def untyped_self_extra_instance(self) -> None:
        if False:
            return 10
        _test_sink(self.tainted_extra_instance)

    def untyped_self_not_tainted(self) -> None:
        if False:
            while True:
                i = 10
        _test_sink(self.not_tainted_class)
        _test_sink(self.not_tainted_instance)
        _test_sink(self.not_tainted_extra_instance)

    def untyped_access_self(self) -> 'Foo':
        if False:
            while True:
                i = 10
        return self

    def typevar_self_class_direct(self: TFoo) -> TFoo:
        if False:
            return 10
        _test_sink(self.__class__.tainted_class)
        return self

    def typevar_self_class(self: TFoo) -> TFoo:
        if False:
            print('Hello World!')
        _test_sink(self.tainted_class)
        return self

    def typevar_self_instance(self: TFoo) -> TFoo:
        if False:
            i = 10
            return i + 15
        _test_sink(self.tainted_instance)
        return self

    def typevar_self_extra_instance(self: TFoo) -> TFoo:
        if False:
            i = 10
            return i + 15
        _test_sink(self.tainted_extra_instance)
        return self

    def typevar_self_not_tainted(self: TFoo) -> TFoo:
        if False:
            for i in range(10):
                print('nop')
        _test_sink(self.not_tainted_class)
        _test_sink(self.not_tainted_instance)
        _test_sink(self.not_tainted_extra_instance)
        return self

    def typevar_access_self(self: TFoo, other: TFoo) -> TFoo:
        if False:
            print('Hello World!')
        return self

    def typevar_access_other(self: TFoo, other: TFoo) -> TFoo:
        if False:
            print('Hello World!')
        return self

    def selftype_self_class_direct(self: Self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        _test_sink(self.__class__.tainted_class)
        return self

    def selftype_self_class(self: Self) -> Self:
        if False:
            print('Hello World!')
        _test_sink(self.tainted_class)
        return self

    def selftype_self_instance(self: Self) -> Self:
        if False:
            print('Hello World!')
        _test_sink(self.tainted_instance)
        return self

    def selftype_self_extra_instance(self: Self) -> Self:
        if False:
            while True:
                i = 10
        _test_sink(self.tainted_extra_instance)
        return self

    def selftype_self_not_tainted(self: Self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        _test_sink(self.not_tainted_class)
        _test_sink(self.not_tainted_instance)
        _test_sink(self.not_tainted_extra_instance)
        return self

    def selftype_access_self(self: Self, other: Self) -> Self:
        if False:
            i = 10
            return i + 15
        return self

    def selftype_access_other(self: Self, other: Self) -> Self:
        if False:
            i = 10
            return i + 15
        return self

    def selftype_access_untyped_self(self, other: Self) -> Self:
        if False:
            i = 10
            return i + 15
        return self

def foo_class_attributes() -> None:
    if False:
        i = 10
        return i + 15
    _test_sink(Foo.tainted_class)
    _test_sink(Foo.not_tainted_class)

def untyped_access_self() -> None:
    if False:
        while True:
            i = 10
    f = Foo('', '')
    _test_sink(f.__class__.tainted_class)
    _test_sink(f.tainted_class)
    _test_sink(f.tainted_instance)
    _test_sink(f.tainted_extra_instance)
    _test_sink(f.__class__.not_tainted_class)
    _test_sink(f.not_tainted_class)
    _test_sink(f.not_tainted_instance)
    _test_sink(f.not_tainted_extra_instance)

def typevar_access_self() -> None:
    if False:
        while True:
            i = 10
    (f1, f2) = (Foo('', ''), Foo('', ''))
    f = f1.typevar_access_self(f2)
    _test_sink(f.__class__.tainted_class)
    _test_sink(f.tainted_class)
    _test_sink(f.tainted_instance)
    _test_sink(f.tainted_extra_instance)
    _test_sink(f.__class__.not_tainted_class)
    _test_sink(f.not_tainted_class)
    _test_sink(f.not_tainted_instance)
    _test_sink(f.not_tainted_extra_instance)

def typevar_access_other() -> None:
    if False:
        while True:
            i = 10
    (f1, f2) = (Foo('', ''), Foo('', ''))
    f = f1.typevar_access_other(f2)
    _test_sink(f.__class__.tainted_class)
    _test_sink(f.tainted_class)
    _test_sink(f.tainted_instance)
    _test_sink(f.tainted_extra_instance)
    _test_sink(f.__class__.not_tainted_class)
    _test_sink(f.not_tainted_class)
    _test_sink(f.not_tainted_instance)
    _test_sink(f.not_tainted_extra_instance)

def selftype_access_self() -> None:
    if False:
        for i in range(10):
            print('nop')
    (f1, f2) = (Foo('', ''), Foo('', ''))
    f = f1.selftype_access_self(f2)
    _test_sink(f.__class__.tainted_class)
    _test_sink(f.tainted_class)
    _test_sink(f.tainted_instance)
    _test_sink(f.tainted_extra_instance)
    _test_sink(f.__class__.not_tainted_class)
    _test_sink(f.not_tainted_class)
    _test_sink(f.not_tainted_instance)
    _test_sink(f.not_tainted_extra_instance)

def selftype_access_other() -> None:
    if False:
        for i in range(10):
            print('nop')
    (f1, f2) = (Foo('', ''), Foo('', ''))
    f = f1.selftype_access_other(f2)
    _test_sink(f.__class__.tainted_class)
    _test_sink(f.tainted_class)
    _test_sink(f.tainted_instance)
    _test_sink(f.tainted_extra_instance)
    _test_sink(f.__class__.not_tainted_class)
    _test_sink(f.not_tainted_class)
    _test_sink(f.not_tainted_instance)
    _test_sink(f.not_tainted_extra_instance)

def selftype_access_untyped_self() -> None:
    if False:
        return 10
    (f1, f2) = (Foo('', ''), Foo('', ''))
    f = f1.selftype_access_untyped_self(f2)
    _test_sink(f.__class__.tainted_class)
    _test_sink(f.tainted_class)
    _test_sink(f.tainted_instance)
    _test_sink(f.tainted_extra_instance)
    _test_sink(f.__class__.not_tainted_class)
    _test_sink(f.not_tainted_class)
    _test_sink(f.not_tainted_instance)
    _test_sink(f.not_tainted_extra_instance)