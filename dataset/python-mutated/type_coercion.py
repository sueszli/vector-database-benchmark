"""Collection of functions to coerce conversion of types with an intelligent guess."""
from collections.abc import Mapping
from itertools import chain
from re import IGNORECASE, compile
from enum import Enum
from ..deprecations import deprecated
from .compat import isiterable
from .decorators import memoizedproperty
from .exceptions import AuxlibError
__all__ = ['boolify', 'typify', 'maybecall', 'listify', 'numberify']
BOOLISH_TRUE = ('true', 'yes', 'on', 'y')
BOOLISH_FALSE = ('false', 'off', 'n', 'no', 'non', 'none', '')
NULL_STRINGS = ('none', '~', 'null', '\x00')
BOOL_COERCEABLE_TYPES = (int, bool, float, complex, list, set, dict, tuple)
NUMBER_TYPES = (int, float, complex)
NUMBER_TYPES_SET = {*NUMBER_TYPES}
STRING_TYPES_SET = {str}
NO_MATCH = object()

class TypeCoercionError(AuxlibError, ValueError):

    def __init__(self, value, msg, *args, **kwargs):
        if False:
            return 10
        self.value = value
        super().__init__(msg, *args, **kwargs)

class _Regex:

    @memoizedproperty
    def BOOLEAN_TRUE(self):
        if False:
            print('Hello World!')
        return (compile('^true$|^yes$|^on$', IGNORECASE), True)

    @memoizedproperty
    def BOOLEAN_FALSE(self):
        if False:
            i = 10
            return i + 15
        return (compile('^false$|^no$|^off$', IGNORECASE), False)

    @memoizedproperty
    def NONE(self):
        if False:
            for i in range(10):
                print('nop')
        return (compile('^none$|^null$', IGNORECASE), None)

    @memoizedproperty
    def INT(self):
        if False:
            return 10
        return (compile('^[-+]?\\d+$'), int)

    @memoizedproperty
    def BIN(self):
        if False:
            return 10
        return (compile('^[-+]?0[bB][01]+$'), bin)

    @memoizedproperty
    def OCT(self):
        if False:
            while True:
                i = 10
        return (compile('^[-+]?0[oO][0-7]+$'), oct)

    @memoizedproperty
    def HEX(self):
        if False:
            for i in range(10):
                print('nop')
        return (compile('^[-+]?0[xX][0-9a-fA-F]+$'), hex)

    @memoizedproperty
    def FLOAT(self):
        if False:
            while True:
                i = 10
        return (compile('^[-+]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][-+]?\\d+)?$'), float)

    @memoizedproperty
    def COMPLEX(self):
        if False:
            for i in range(10):
                print('nop')
        return (compile('^(?:[-+]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][-+]?\\d+)?)?[-+]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][-+]?\\d+)?j$'), complex)

    @property
    def numbers(self):
        if False:
            while True:
                i = 10
        yield self.INT
        yield self.FLOAT
        yield self.BIN
        yield self.OCT
        yield self.HEX
        yield self.COMPLEX

    @property
    def boolean(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.BOOLEAN_TRUE
        yield self.BOOLEAN_FALSE

    @property
    def none(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.NONE

    def convert_number(self, value_string):
        if False:
            i = 10
            return i + 15
        return self._convert(value_string, (self.numbers,))

    def convert(self, value_string):
        if False:
            for i in range(10):
                print('nop')
        return self._convert(value_string, (self.boolean, self.none, self.numbers))

    def _convert(self, value_string, type_list):
        if False:
            print('Hello World!')
        return next((typish(value_string) if callable(typish) else typish for (regex, typish) in chain.from_iterable(type_list) if regex.match(value_string)), NO_MATCH)
_REGEX = _Regex()

def numberify(value):
    if False:
        while True:
            i = 10
    "\n\n    Examples:\n        >>> [numberify(x) for x in ('1234', 1234, '0755', 0o0755, False, 0, '0', True, 1, '1')]\n          [1234, 1234, 755, 493, 0, 0, 0, 1, 1, 1]\n        >>> [numberify(x) for x in ('12.34', 12.34, 1.2+3.5j, '1.2+3.5j')]\n        [12.34, 12.34, (1.2+3.5j), (1.2+3.5j)]\n\n    "
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, NUMBER_TYPES):
        return value
    candidate = _REGEX.convert_number(value)
    if candidate is not NO_MATCH:
        return candidate
    raise TypeCoercionError(value, f'Cannot convert {value} to a number.')

def boolify(value, nullable=False, return_string=False):
    if False:
        for i in range(10):
            print('nop')
    'Convert a number, string, or sequence type into a pure boolean.\n\n    Args:\n        value (number, string, sequence): pretty much anything\n\n    Returns:\n        bool: boolean representation of the given value\n\n    Examples:\n        >>> [boolify(x) for x in (\'yes\', \'no\')]\n        [True, False]\n        >>> [boolify(x) for x in (0.1, 0+0j, True, \'0\', \'0.0\', \'0.1\', \'2\')]\n        [True, False, True, False, False, True, True]\n        >>> [boolify(x) for x in ("true", "yes", "on", "y")]\n        [True, True, True, True]\n        >>> [boolify(x) for x in ("no", "non", "none", "off", "")]\n        [False, False, False, False, False]\n        >>> [boolify(x) for x in ([], set(), dict(), tuple())]\n        [False, False, False, False]\n        >>> [boolify(x) for x in ([1], set([False]), dict({\'a\': 1}), tuple([2]))]\n        [True, True, True, True]\n\n    '
    if isinstance(value, BOOL_COERCEABLE_TYPES):
        return bool(value)
    val = str(value).strip().lower().replace('.', '', 1)
    if val.isnumeric():
        return bool(float(val))
    elif val in BOOLISH_TRUE:
        return True
    elif nullable and val in NULL_STRINGS:
        return None
    elif val in BOOLISH_FALSE:
        return False
    else:
        try:
            return bool(complex(val))
        except ValueError:
            if isinstance(value, str) and return_string:
                return value
            raise TypeCoercionError(value, 'The value %r cannot be boolified.' % value)

@deprecated('24.3', '24.9')
def boolify_truthy_string_ok(value):
    if False:
        print('Hello World!')
    try:
        return boolify(value)
    except ValueError:
        assert isinstance(value, str), repr(value)
        return True

def typify_str_no_hint(value):
    if False:
        print('Hello World!')
    candidate = _REGEX.convert(value)
    return candidate if candidate is not NO_MATCH else value

def typify(value, type_hint=None):
    if False:
        i = 10
        return i + 15
    "Take a primitive value, usually a string, and try to make a more relevant type out of it.\n    An optional type_hint will try to coerce the value to that type.\n\n    Args:\n        value (Any): Usually a string, not a sequence\n        type_hint (type or tuple[type]):\n\n    Examples:\n        >>> typify('32')\n        32\n        >>> typify('32', float)\n        32.0\n        >>> typify('32.0')\n        32.0\n        >>> typify('32.0.0')\n        '32.0.0'\n        >>> [typify(x) for x in ('true', 'yes', 'on')]\n        [True, True, True]\n        >>> [typify(x) for x in ('no', 'FALSe', 'off')]\n        [False, False, False]\n        >>> [typify(x) for x in ('none', 'None', None)]\n        [None, None, None]\n\n    "
    if isinstance(value, str):
        value = value.strip()
    elif type_hint is None:
        return value
    if isiterable(type_hint):
        if isinstance(type_hint, type) and issubclass(type_hint, Enum):
            try:
                return type_hint(value)
            except ValueError as e:
                try:
                    return type_hint[value]
                except KeyError:
                    raise TypeCoercionError(value, str(e))
        type_hint = set(type_hint)
        if not type_hint - NUMBER_TYPES_SET:
            return numberify(value)
        elif not type_hint - STRING_TYPES_SET:
            return str(value)
        elif not type_hint - {bool, type(None)}:
            return boolify(value, nullable=True)
        elif not type_hint - (STRING_TYPES_SET | {bool}):
            return boolify(value, return_string=True)
        elif not type_hint - (STRING_TYPES_SET | {type(None)}):
            value = str(value)
            return None if value.lower() == 'none' else value
        elif not type_hint - {bool, int}:
            return typify_str_no_hint(str(value))
        else:
            raise NotImplementedError()
    elif type_hint is not None:
        try:
            return boolify(value) if type_hint == bool else type_hint(value)
        except ValueError as e:
            raise TypeCoercionError(value, str(e))
    else:
        return typify_str_no_hint(value)

def typify_data_structure(value, type_hint=None):
    if False:
        while True:
            i = 10
    if isinstance(value, Mapping):
        return type(value)(((k, typify(v, type_hint)) for (k, v) in value.items()))
    elif isiterable(value):
        return type(value)((typify(v, type_hint) for v in value))
    elif isinstance(value, str) and isinstance(type_hint, type) and issubclass(type_hint, str):
        return type_hint(value)
    else:
        return typify(value, type_hint)

def maybecall(value):
    if False:
        i = 10
        return i + 15
    return value() if callable(value) else value

@deprecated('24.3', '24.9')
def listify(val, return_type=tuple):
    if False:
        print('Hello World!')
    "\n    Examples:\n        >>> listify('abc', return_type=list)\n        ['abc']\n        >>> listify(None)\n        ()\n        >>> listify(False)\n        (False,)\n        >>> listify(('a', 'b', 'c'), return_type=list)\n        ['a', 'b', 'c']\n    "
    if val is None:
        return return_type()
    elif isiterable(val):
        return return_type(val)
    else:
        return return_type((val,))