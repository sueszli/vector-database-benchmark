from collections.abc import Iterable, Mapping
from collections import UserString
from io import IOBase
from os import PathLike
from typing import Union, TypedDict, TypeVar
try:
    from types import UnionType
except ImportError:
    UnionType = ()
try:
    from typing_extensions import TypedDict as ExtTypedDict
except ImportError:
    ExtTypedDict = None
TRUE_STRINGS = {'TRUE', 'YES', 'ON', '1'}
FALSE_STRINGS = {'FALSE', 'NO', 'OFF', '0', 'NONE', ''}
typeddict_types = (type(TypedDict('Dummy', {})),)
if ExtTypedDict:
    typeddict_types += (type(ExtTypedDict('Dummy', {})),)

def is_integer(item):
    if False:
        i = 10
        return i + 15
    return isinstance(item, int)

def is_number(item):
    if False:
        return 10
    return isinstance(item, (int, float))

def is_bytes(item):
    if False:
        return 10
    return isinstance(item, (bytes, bytearray))

def is_string(item):
    if False:
        print('Hello World!')
    return isinstance(item, str)

def is_pathlike(item):
    if False:
        return 10
    return isinstance(item, PathLike)

def is_list_like(item):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(item, (str, bytes, bytearray, UserString, IOBase)):
        return False
    return isinstance(item, Iterable)

def is_dict_like(item):
    if False:
        return 10
    return isinstance(item, Mapping)

def is_union(item):
    if False:
        while True:
            i = 10
    return isinstance(item, UnionType) or getattr(item, '__origin__', None) is Union

def type_name(item, capitalize=False):
    if False:
        print('Hello World!')
    'Return "non-technical" type name for objects and types.\n\n    For example, \'integer\' instead of \'int\' and \'file\' instead of \'TextIOWrapper\'.\n    '
    if getattr(item, '__origin__', None):
        item = item.__origin__
    if hasattr(item, '_name') and item._name:
        name = item._name
    elif is_union(item):
        name = 'Union'
    elif isinstance(item, IOBase):
        name = 'file'
    else:
        typ = type(item) if not isinstance(item, type) else item
        named_types = {str: 'string', bool: 'boolean', int: 'integer', type(None): 'None', dict: 'dictionary'}
        name = named_types.get(typ, typ.__name__.strip('_'))
    return name.capitalize() if capitalize and name.islower() else name

def type_repr(typ, nested=True):
    if False:
        print('Hello World!')
    "Return string representation for types.\n\n    Aims to look as much as the source code as possible. For example, 'List[Any]'\n    instead of 'typing.List[typing.Any]'.\n    "
    if typ is type(None):
        return 'None'
    if typ is Ellipsis:
        return '...'
    if is_union(typ):
        return ' | '.join((type_repr(a) for a in typ.__args__)) if nested else 'Union'
    name = _get_type_name(typ)
    if nested and has_args(typ):
        args = ', '.join((type_repr(a) for a in typ.__args__))
        return f'{name}[{args}]'
    return name

def _get_type_name(typ):
    if False:
        return 10
    for attr in ('__name__', '_name'):
        name = getattr(typ, attr, None)
        if name:
            return name
    return str(typ)

def has_args(type):
    if False:
        for i in range(10):
            print('nop')
    "Helper to check has type valid ``__args__``.\n\n   ``__args__`` contains TypeVars when accessed directly from ``typing.List`` and\n   other such types with Python 3.8. Python 3.9+ don't have ``__args__`` at all.\n   Parameterize usages like ``List[int].__args__`` always work the same way.\n\n    This helper can be removed in favor of using ``hasattr(type, '__args__')``\n    when we support only Python 3.9 and newer.\n    "
    args = getattr(type, '__args__', None)
    return bool(args and (not all((isinstance(a, TypeVar) for a in args))))

def is_truthy(item):
    if False:
        i = 10
        return i + 15
    "Returns `True` or `False` depending on is the item considered true or not.\n\n    Validation rules:\n\n    - If the value is a string, it is considered false if it is `'FALSE'`,\n      `'NO'`, `'OFF'`, `'0'`, `'NONE'` or `''`, case-insensitively.\n    - Other strings are considered true.\n    - Other values are handled by using the standard `bool()` function.\n\n    Designed to be used also by external test libraries that want to handle\n    Boolean values similarly as Robot Framework itself. See also\n    :func:`is_falsy`.\n    "
    if is_string(item):
        return item.upper() not in FALSE_STRINGS
    return bool(item)

def is_falsy(item):
    if False:
        i = 10
        return i + 15
    'Opposite of :func:`is_truthy`.'
    return not is_truthy(item)