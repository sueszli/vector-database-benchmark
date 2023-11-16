"""Utilities for accessing Python generic type annotations (typing.*)."""
import collections.abc
import typing

def is_generic_union(tp):
    if False:
        i = 10
        return i + 15
    'Returns true if `tp` is a parameterized typing.Union value.'
    return tp is not typing.Union and getattr(tp, '__origin__', None) is typing.Union

def is_generic_tuple(tp):
    if False:
        i = 10
        return i + 15
    'Returns true if `tp` is a parameterized typing.Tuple value.'
    return tp not in (tuple, typing.Tuple) and getattr(tp, '__origin__', None) in (tuple, typing.Tuple)

def is_generic_list(tp):
    if False:
        i = 10
        return i + 15
    'Returns true if `tp` is a parameterized typing.List value.'
    return tp not in (list, typing.List) and getattr(tp, '__origin__', None) in (list, typing.List)

def is_generic_mapping(tp):
    if False:
        print('Hello World!')
    'Returns true if `tp` is a parameterized typing.Mapping value.'
    return tp not in (collections.abc.Mapping, typing.Mapping) and getattr(tp, '__origin__', None) in (collections.abc.Mapping, typing.Mapping)

def is_forward_ref(tp):
    if False:
        return 10
    'Returns true if `tp` is a typing forward reference.'
    if hasattr(typing, 'ForwardRef'):
        return isinstance(tp, typing.ForwardRef)
    elif hasattr(typing, '_ForwardRef'):
        return isinstance(tp, typing._ForwardRef)
    else:
        return False
if hasattr(typing, 'get_args'):
    get_generic_type_args = typing.get_args
else:
    get_generic_type_args = lambda tp: tp.__args__