from datetime import datetime, date, timedelta
from decimal import Decimal

def integer(argument: 'Integer', expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def int_(argument: 'INT', expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def long_(argument: 'lOnG', expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def float_(argument: 'Float', expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

def double(argument: 'Double', expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def decimal(argument: 'DECIMAL', expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def boolean(argument: 'Boolean', expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def bool_(argument: 'Bool', expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

def string(argument: 'String', expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

def bytes_(argument: 'BYTES', expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def bytearray_(argument: 'ByteArray', expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

def datetime_(argument: 'DateTime', expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def date_(argument: 'Date', expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

def timedelta_(argument: 'TimeDelta', expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

def list_(argument: 'List', expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def tuple_(argument: 'TUPLE', expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def dictionary(argument: 'Dictionary', expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

def dict_(argument: 'Dict', expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def map_(argument: 'Map', expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def set_(argument: 'Set', expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

def frozenset_(argument: 'FrozenSet', expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

def _validate_type(argument, expected):
    if False:
        i = 10
        return i + 15
    if isinstance(expected, str):
        expected = eval(expected)
    if argument != expected or type(argument) != type(expected):
        raise AssertionError('%r (%s) != %r (%s)' % (argument, type(argument).__name__, expected, type(expected).__name__))