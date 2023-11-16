"""This file contains the typing api that should exist in python in
order to do metaprogramming and reflection on the built-in typing module.
"""
import typing
from typing_extensions import get_args, get_origin
import dagster._check as check

def is_closed_python_optional_type(ttype):
    if False:
        print('Hello World!')
    origin = get_origin(ttype)
    args = get_args(ttype)
    return origin is typing.Union and len(args) == 2 and (args[1] is type(None))

def is_python_dict_type(ttype):
    if False:
        for i in range(10):
            print('nop')
    origin = get_origin(ttype)
    return ttype is dict or origin is dict

def is_closed_python_list_type(ttype):
    if False:
        while True:
            i = 10
    origin = get_origin(ttype)
    args = get_args(ttype)
    return origin is list and args != () and (type(args[0]) != typing.TypeVar)

def is_closed_python_dict_type(ttype):
    if False:
        for i in range(10):
            print('nop')
    'A "closed" generic type has all of its type parameters parameterized\n    by other closed or concrete types.\n\n    e.g.\n\n    Returns true for typing.Dict[int, str] but not for typing.Dict.\n\n    Tests document current behavior (not recursive) -- i.e., typing.Dict[str, Dict] returns True.\n    '
    origin = get_origin(ttype)
    args = get_args(ttype)
    return origin is dict and args != () and (type(args[0]) != typing.TypeVar) and (type(args[1]) != typing.TypeVar)

def is_closed_python_tuple_type(ttype):
    if False:
        while True:
            i = 10
    'A "closed" generic type has all of its type parameters parameterized\n    by other closed or concrete types.\n\n    e.g.\n\n    Returns true for Tuple[int] or Tuple[str, int] but false for Tuple or tuple\n    '
    origin = get_origin(ttype)
    args = get_args(ttype)
    return origin is tuple and args != ()

def is_closed_python_set_type(ttype):
    if False:
        for i in range(10):
            print('nop')
    'A "closed" generic type has all of its type parameters parameterized\n    by other closed or concrete types.\n\n    e.g.\n\n    Returns true for Set[string] but false for Set or set\n    '
    origin = get_origin(ttype)
    args = get_args(ttype)
    return origin is set and args != () and (type(args[0]) != typing.TypeVar)

def get_optional_inner_type(ttype):
    if False:
        while True:
            i = 10
    check.invariant(is_closed_python_optional_type(ttype), 'type must pass is_closed_python_optional_type check')
    return get_args(ttype)[0]

def get_list_inner_type(ttype):
    if False:
        print('Hello World!')
    check.param_invariant(is_closed_python_list_type(ttype), 'ttype')
    return get_args(ttype)[0]

def get_set_inner_type(ttype):
    if False:
        i = 10
        return i + 15
    check.param_invariant(is_closed_python_set_type(ttype), 'ttype')
    return get_args(ttype)[0]

def get_tuple_type_params(ttype):
    if False:
        return 10
    check.param_invariant(is_closed_python_tuple_type(ttype), 'ttype')
    return get_args(ttype)

def get_dict_key_value_types(ttype):
    if False:
        for i in range(10):
            print('nop')
    check.param_invariant(is_closed_python_dict_type(ttype), 'ttype')
    return get_args(ttype)

def is_typing_type(ttype):
    if False:
        for i in range(10):
            print('nop')
    return is_closed_python_dict_type(ttype) or is_closed_python_optional_type(ttype) or is_closed_python_set_type(ttype) or is_closed_python_tuple_type(ttype) or is_closed_python_list_type(ttype) or (ttype is typing.Tuple) or (ttype is typing.Set) or (ttype is typing.Dict) or (ttype is typing.List)