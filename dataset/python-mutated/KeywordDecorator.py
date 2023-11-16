from collections import abc
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Flag, Enum, IntFlag, IntEnum
from fractions import Fraction
from numbers import Integral, Real
from os import PathLike
from pathlib import Path, PurePath
from typing import Union
from robot.api.deco import keyword

class MyEnum(Enum):
    FOO = 1
    bar = 'xxx'
    foo = 'yyy'
    normalize_me = True

class MyFlag(Flag):
    RED = 1
    BLUE = 2

class MyIntEnum(IntEnum):
    ON = 1
    OFF = 0

class MyIntFlag(IntFlag):
    R = 4
    W = 2
    X = 1

class Unknown:
    pass

@keyword(types={'argument': int})
def integer(argument, expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

@keyword(types={'argument': Integral})
def integral(argument, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

@keyword(types={'argument': float})
def float_(argument, expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

@keyword(types={'argument': Real})
def real(argument, expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

@keyword(types={'argument': Decimal})
def decimal(argument, expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

@keyword(types={'argument': bool})
def boolean(argument, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

@keyword(types={'argument': str})
def string(argument, expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

@keyword(types={'argument': bytes})
def bytes_(argument, expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

@keyword(types={'argument': getattr(abc, 'ByteString', None)})
def bytestring(argument, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

@keyword(types={'argument': bytearray})
def bytearray_(argument, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

@keyword(types={'argument': datetime})
def datetime_(argument, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

@keyword(types={'argument': date})
def date_(argument, expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

@keyword(types={'argument': timedelta})
def timedelta_(argument, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

@keyword(types={'argument': Path})
def path(argument, expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

@keyword(types={'argument': PurePath})
def pure_path(argument, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

@keyword(types={'argument': PathLike})
def path_like(argument, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

@keyword(types={'argument': MyEnum})
def enum(argument, expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

@keyword(types={'argument': MyFlag})
def flag(argument, expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

@keyword(types=[MyIntEnum])
def int_enum(argument, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

@keyword(types=[MyIntFlag])
def int_flag(argument, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

@keyword(types={'argument': type(None)})
def nonetype(argument, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

@keyword(types={'argument': None})
def none(argument, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

@keyword(types={'argument': list})
def list_(argument, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

@keyword(types={'argument': abc.Sequence})
def sequence(argument, expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

@keyword(types={'argument': abc.MutableSequence})
def mutable_sequence(argument, expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

@keyword(types={'argument': tuple})
def tuple_(argument, expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

@keyword(types={'argument': dict})
def dictionary(argument, expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

@keyword(types={'argument': abc.Mapping})
def mapping(argument, expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

@keyword(types={'argument': abc.MutableMapping})
def mutable_mapping(argument, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

@keyword(types={'argument': set})
def set_(argument, expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

@keyword(types={'argument': abc.Set})
def set_abc(argument, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

@keyword(types={'argument': abc.MutableSet})
def mutable_set(argument, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

@keyword(types={'argument': frozenset})
def frozenset_(argument, expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

@keyword(types={'argument': Unknown})
def unknown(argument, expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

@keyword(types={'argument': 'this is just a random string'})
def non_type(argument, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

@keyword(types={'argument': int})
def varargs(*argument, **expected):
    if False:
        for i in range(10):
            print('nop')
    expected = expected.pop('expected', None)
    _validate_type(argument, expected)

@keyword(types={'argument': int})
def kwargs(expected=None, **argument):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

@keyword(types={'argument': float})
def kwonly(*, argument, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

@keyword(types='invalid')
def invalid_type_spec():
    if False:
        print('Hello World!')
    raise RuntimeError('Should not be executed')

@keyword(types={'no_match': int, 'xxx': 42})
def non_matching_name(argument):
    if False:
        for i in range(10):
            print('nop')
    raise RuntimeError('Should not be executed')

@keyword(types={'argument': int, 'return': float})
def return_type(argument, expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

@keyword(types=[list])
def type_and_default_1(argument=None, expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

@keyword(types=[int])
def type_and_default_2(argument=True, expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

@keyword(types=[timedelta])
def type_and_default_3(argument=0, expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

@keyword(types={'argument': Union[int, None, float]})
def multiple_types_using_union(argument, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

@keyword(types={'argument': (int, None, float)})
def multiple_types_using_tuple(argument, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def _validate_type(argument, expected):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(expected, str):
        expected = eval(expected)
    if argument != expected or type(argument) != type(expected):
        raise AssertionError('%r (%s) != %r (%s)' % (argument, type(argument).__name__, expected, type(expected).__name__))