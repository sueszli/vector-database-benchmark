"""Utilities to generate some basic types."""
from typing import Tuple
from pytype.pytd import pytd
_STRING_TYPES = ('str', 'bytes', 'unicode')
_ParametersType = Tuple[pytd.Type, ...]

def pytd_list(typ: str) -> pytd.Type:
    if False:
        for i in range(10):
            print('nop')
    if typ:
        return pytd.GenericType(pytd.NamedType('typing.List'), (pytd.NamedType(typ),))
    else:
        return pytd.NamedType('typing.List')

def is_any(val) -> bool:
    if False:
        i = 10
        return i + 15
    if isinstance(val, pytd.AnythingType):
        return True
    elif isinstance(val, pytd.NamedType):
        return val.name == 'typing.Any'
    else:
        return False

def is_none(t) -> bool:
    if False:
        i = 10
        return i + 15
    return isinstance(t, pytd.NamedType) and t.name in ('None', 'NoneType')

def heterogeneous_tuple(base_type: pytd.NamedType, parameters: _ParametersType) -> pytd.Type:
    if False:
        i = 10
        return i + 15
    return pytd.TupleType(base_type=base_type, parameters=parameters)

def pytd_type(value: pytd.Type) -> pytd.Type:
    if False:
        i = 10
        return i + 15
    return pytd.GenericType(pytd.NamedType('type'), (value,))

def pytd_callable(base_type: pytd.NamedType, parameters: _ParametersType, arg_is_paramspec: bool=False) -> pytd.Type:
    if False:
        for i in range(10):
            print('nop')
    'Create a pytd.CallableType.'
    if len(parameters) != 2:
        raise TypeError(f'Expected 2 parameters to Callable, got {len(parameters)}')
    (args, ret) = parameters
    if isinstance(args, list):
        if not args or args == [pytd.NothingType()]:
            parameters = (ret,)
        else:
            if any((x.__class__.__name__ == 'Ellipsis' for x in args)):
                if is_any(ret):
                    ret = 'Any'
                msg = f'Invalid Callable args, did you mean Callable[..., {ret}]?'
                raise TypeError(msg)
            parameters = tuple(args) + (ret,)
        return pytd.CallableType(base_type=base_type, parameters=parameters)
    elif arg_is_paramspec or isinstance(args, pytd.Concatenate):
        return pytd.CallableType(base_type=base_type, parameters=parameters)
    else:
        if not is_any(args):
            msg = 'First argument to Callable must be a list of argument types (got %r)' % args
            raise TypeError(msg)
        return pytd.GenericType(base_type=base_type, parameters=parameters)