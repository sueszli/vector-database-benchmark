"""Class definitions in pyi files."""
import ast as astlib
from typing import cast, Callable, List
from pytype.pyi import types
from pytype.pytd import pytd
_ParseError = types.ParseError

def get_bases(bases: List[pytd.Type], type_match: Callable[..., bool]) -> List[pytd.Type]:
    if False:
        print('Hello World!')
    'Collect base classes.'
    bases_out = []
    namedtuple_index = None
    for (i, p) in enumerate(bases):
        if p.name and type_match(p.name, 'typing.Protocol'):
            if isinstance(p, pytd.GenericType):
                bases_out.append(p.Replace(base_type=pytd.NamedType('typing.Generic')))
            bases_out.append(pytd.NamedType('typing.Protocol'))
        elif isinstance(p, pytd.NamedType) and p.name == 'typing.NamedTuple':
            if namedtuple_index is not None:
                raise _ParseError('cannot inherit from bare NamedTuple more than once')
            namedtuple_index = i
            bases_out.append(p)
        elif isinstance(p, pytd.Type):
            bases_out.append(p)
        else:
            msg = f'Unexpected class base: {p}'
            raise _ParseError(msg)
    return bases_out

def get_keywords(keywords: List[astlib.keyword]):
    if False:
        return 10
    'Get valid class keywords.'
    valid_keywords = []
    for k in keywords:
        (keyword, value) = (k.arg, k.value)
        if keyword not in ('metaclass', 'total'):
            raise _ParseError(f'Unexpected classdef kwarg {keyword!r}')
        if isinstance(value, types.Pyval):
            pytd_value = value.to_pytd_literal()
        else:
            pytd_value = cast(pytd.Type, value)
        valid_keywords.append((keyword, pytd_value))
    return valid_keywords