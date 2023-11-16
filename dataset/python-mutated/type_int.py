from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import numpy as np
import sympy as sm
import math
import logging
from .annotate import class_annotate, annotate, delay_type
from .type_bool import bool
from .type_spec import Type

def make_int(width, unsigned):
    if False:
        for i in range(10):
            print('nop')
    delay_type_int = getattr(delay_type, unsigned + 'int' + str(width))

    @class_annotate()
    class int:
        _width = width
        _unsigned = unsigned

        @annotate(v=delay_type_int)
        def __init__(self, v=0):
            if False:
                for i in range(10):
                    print('nop')
            self._val = v

        @property
        def val(self):
            if False:
                for i in range(10):
                    print('nop')
            return self._val

        @val.setter
        def val(self, v):
            if False:
                for i in range(10):
                    print('nop')
            from .type_mapping import nptype_from_builtin, builtin_to_string, numpy_type_to_builtin_type
            if not isinstance(v, (np.generic, sm.Basic)):
                raise ValueError('types should have value of numpy type or Symbols, got {} instead'.format(type(v)))
            if isinstance(v, sm.Basic):
                self._val = v
            elif isinstance(v, np.integer):
                v_type = numpy_type_to_builtin_type(v.dtype)
                if v_type.get_bitwidth() <= self.get_bitwidth() and (v >= 0 or (v < 0 and (not self.is_unsigned()))):
                    self._val = v
                else:
                    self._val = v.astype(nptype_from_builtin(self.__class__))
                    logging.warning('Saving value type of {} into a builtin type of {}, might overflow or loses precision!'.format(v.dtype, builtin_to_string(self.__class__)))
            else:
                self._val = v.astype(nptype_from_builtin(self.__class__))
                logging.warning('Saving value type of {} into a builtin type of {}, might be incompatible or loses precision!'.format(v.dtype, builtin_to_string(self.__class__)))

        @classmethod
        def __type_info__(cls):
            if False:
                return 10
            return Type(cls._unsigned + 'int' + str(cls._width), python_class=cls)

        @classmethod
        def get_bitwidth(cls):
            if False:
                for i in range(10):
                    print('nop')
            return cls._width

        @classmethod
        def is_unsigned(cls):
            if False:
                i = 10
                return i + 15
            return cls._unsigned == 'u'

        @annotate(delay_type_int, other=delay_type_int)
        def __add__(self, other):
            if False:
                print('Hello World!')
            assert isinstance(other, int)
            return int(self.val + other.val)

        @annotate(delay_type_int, other=delay_type_int)
        def __sub__(self, other):
            if False:
                print('Hello World!')
            assert isinstance(other, int)
            return int(self.val - other.val)

        @annotate(delay_type_int, other=delay_type_int)
        def __mul__(self, other):
            if False:
                return 10
            assert isinstance(other, int)
            return int(self.val * other.val)

        @annotate(delay_type_int, other=delay_type_int)
        def __div__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            assert isinstance(other, int)
            return int(self.val // other.val)

        @annotate(delay_type_int, other=delay_type_int)
        def __mod__(self, other):
            if False:
                while True:
                    i = 10
            assert isinstance(other, int)
            return int(self.val % other.val)

        @annotate(delay_type.bool, other=delay_type_int)
        def __lt__(self, other):
            if False:
                print('Hello World!')
            return bool(self.val < other.val)

        @annotate(delay_type.bool, other=delay_type_int)
        def __gt__(self, other):
            if False:
                i = 10
                return i + 15
            return bool(self.val > other.val)

        @annotate(delay_type.bool, other=delay_type_int)
        def __le__(self, other):
            if False:
                print('Hello World!')
            return bool(self.val <= other.val)

        @annotate(delay_type.bool, other=delay_type_int)
        def __ge__(self, other):
            if False:
                return 10
            return bool(self.val >= other.val)

        @annotate(delay_type.bool, other=delay_type_int)
        def __eq__(self, other):
            if False:
                print('Hello World!')
            return bool(self.val == other.val)

        @annotate(delay_type.bool, other=delay_type_int)
        def __ne__(self, other):
            if False:
                return 10
            return bool(self.val != other.val)

        @annotate(delay_type.bool)
        def __bool__(self):
            if False:
                print('Hello World!')
            return self.val != 0

        @annotate(delay_type_int)
        def __int__(self):
            if False:
                i = 10
                return i + 15
            return int(self)

        @annotate(delay_type.double)
        def __double__(self):
            if False:
                while True:
                    i = 10
            return float(self.val)

        @annotate(delay_type.str)
        def __str__(self):
            if False:
                return 10
            return str(self.val)

        @annotate(delay_type.double)
        def __log__(self):
            if False:
                return 10
            return math.log(self.val)

        @annotate(delay_type.double)
        def __exp__(self):
            if False:
                while True:
                    i = 10
            return math.exp(self.val)

        @annotate(delay_type_int)
        def __neg__(self):
            if False:
                for i in range(10):
                    print('nop')
            return int(-self.val)
    return int
int8 = make_int(8, '')
int16 = make_int(16, '')
int32 = make_int(32, '')
int64 = make_int(64, '')
int = int64
uint8 = make_int(8, 'u')
uint16 = make_int(16, 'u')
uint32 = make_int(32, 'u')
uint64 = make_int(64, 'u')
uint = uint64

def is_int(t):
    if False:
        print('Hello World!')
    return any((t is i or isinstance(t, i) for i in [int8, int16, int32, int64, uint8, uint16, uint32, uint64]))