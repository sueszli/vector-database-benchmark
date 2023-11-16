from numbers import Rational
from typing import TypedDict

class MyObject:
    pass

class AnotherObject:
    pass

class BadRationalMeta(type(Rational)):

    def __instancecheck__(self, instance):
        if False:
            print('Hello World!')
        raise TypeError('Bang!')

class BadRational(Rational, metaclass=BadRationalMeta):
    pass

def create_my_object():
    if False:
        for i in range(10):
            print('nop')
    return MyObject()

def union_of_int_float_and_string(argument: int | float | str, expected):
    if False:
        return 10
    assert argument == expected

def union_of_int_and_float(argument: int | float, expected=object()):
    if False:
        return 10
    assert argument == expected

def union_with_int_and_none(argument: int | None, expected=object()):
    if False:
        i = 10
        return i + 15
    assert argument == expected

def union_with_int_none_and_str(argument: int | None | str, expected):
    if False:
        while True:
            i = 10
    assert argument == expected

def union_with_abc(argument: Rational | None, expected):
    if False:
        while True:
            i = 10
    assert argument == expected

def union_with_str_and_abc(argument: str | Rational, expected):
    if False:
        return 10
    assert argument == expected

def union_with_subscripted_generics(argument: list[int] | int, expected=object()):
    if False:
        print('Hello World!')
    assert argument == eval(expected), '%r != %s' % (argument, expected)

def union_with_subscripted_generics_and_str(argument: list[str] | str, expected):
    if False:
        while True:
            i = 10
    assert argument == eval(expected), '%r != %s' % (argument, expected)

def union_with_typeddict(argument: TypedDict('X', x=int) | None, expected):
    if False:
        while True:
            i = 10
    assert argument == eval(expected), '%r != %s' % (argument, expected)

def union_with_item_not_liking_isinstance(argument: BadRational | bool, expected):
    if False:
        print('Hello World!')
    assert argument == expected, '%r != %r' % (argument, expected)

def custom_type_in_union(argument: MyObject | str, expected_type):
    if False:
        for i in range(10):
            print('nop')
    assert type(argument).__name__ == expected_type

def only_custom_types_in_union(argument: MyObject | AnotherObject, expected_type):
    if False:
        for i in range(10):
            print('nop')
    assert type(argument).__name__ == expected_type

def union_with_string_first(argument: str | None, expected):
    if False:
        for i in range(10):
            print('nop')
    assert argument == expected