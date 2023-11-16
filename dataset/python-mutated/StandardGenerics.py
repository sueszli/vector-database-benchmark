from typing import Union

class Unknown:
    pass

def list_(argument: list[int], expected=None, same=False):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected, same)

def list_with_unknown(argument: list[Unknown], expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def list_in_union_1(argument: Union[str, list[str]], expected=None, same=False):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected, same)

def list_in_union_2(argument: Union[list[str], str], expected=None, same=False):
    if False:
        return 10
    _validate_type(argument, expected, same)

def tuple_(argument: tuple[int, bool, float], expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def tuple_with_unknown(argument: tuple[Unknown, int], expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def tuple_in_union_1(argument: Union[str, tuple[str, str, str]], expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

def tuple_in_union_2(argument: Union[tuple[str, str, str], str], expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def homogenous_tuple(argument: tuple[int, ...], expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def homogenous_tuple_with_unknown(argument: tuple[Unknown, ...], expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def homogenous_tuple_in_union_1(argument: Union[str, tuple[str, ...]], expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def homogenous_tuple_in_union_2(argument: Union[tuple[str, ...], str], expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def dict_(argument: dict[int, float], expected=None, same=False):
    if False:
        print('Hello World!')
    _validate_type(argument, expected, same)

def dict_with_unknown_key(argument: dict[Unknown, int], expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

def dict_with_unknown_value(argument: dict[int, Unknown], expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

def dict_in_union_1(argument: Union[str, dict[str, str]], expected=None, same=False):
    if False:
        return 10
    _validate_type(argument, expected, same)

def dict_in_union_2(argument: Union[dict[str, str], str], expected=None, same=False):
    if False:
        print('Hello World!')
    _validate_type(argument, expected, same)

def set_(argument: set[bool], expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def set_with_unknown(argument: set[Unknown], expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def set_in_union_1(argument: Union[str, set[str]], expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def set_in_union_2(argument: Union[set[str], str], expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

def nested_generics(argument: list[tuple[int, int]], expected=None, same=False):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected, same)

def invalid_list(a: list[int, float]):
    if False:
        i = 10
        return i + 15
    pass

def invalid_tuple(a: tuple[int, float, ...]):
    if False:
        print('Hello World!')
    pass

def invalid_dict(a: dict[int]):
    if False:
        return 10
    pass

def invalid_set(a: set[int, float]):
    if False:
        print('Hello World!')
    pass

def _validate_type(argument, expected, same=False):
    if False:
        while True:
            i = 10
    if isinstance(expected, str):
        expected = eval(expected)
    if argument != expected or type(argument) != type(expected):
        atype = type(argument).__name__
        etype = type(expected).__name__
        raise AssertionError(f'{argument!r} ({atype}) != {expected!r} ({etype})')
    if same and argument is not expected:
        raise AssertionError(f'{argument} (id: {id(argument)}) is not same as {expected} (id: {id(expected)})')