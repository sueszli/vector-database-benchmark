"""Implementation of :class:`ModularInteger` class. """
from __future__ import annotations
from typing import Any
import operator
from sympy.polys.polyutils import PicklableWithSlots
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.domainelement import DomainElement
from sympy.utilities import public
from sympy.utilities.exceptions import sympy_deprecation_warning

@public
class ModularInteger(PicklableWithSlots, DomainElement):
    """A class representing a modular integer. """
    (mod, dom, sym, _parent) = (None, None, None, None)
    __slots__ = ('val',)

    def parent(self):
        if False:
            return 10
        return self._parent

    def __init__(self, val):
        if False:
            return 10
        if isinstance(val, self.__class__):
            self.val = val.val % self.mod
        else:
            self.val = self.dom.convert(val) % self.mod

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((self.val, self.mod))

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s(%s)' % (self.__class__.__name__, self.val)

    def __str__(self):
        if False:
            print('Hello World!')
        return '%s mod %s' % (self.val, self.mod)

    def __int__(self):
        if False:
            print('Hello World!')
        return int(self.val)

    def to_int(self):
        if False:
            i = 10
            return i + 15
        sympy_deprecation_warning('ModularInteger.to_int() is deprecated.\n\n            Use int(a) or K = GF(p) and K.to_int(a) instead of a.to_int().\n            ', deprecated_since_version='1.13', active_deprecations_target='modularinteger-to-int')
        if self.sym:
            if self.val <= self.mod // 2:
                return self.val
            else:
                return self.val - self.mod
        else:
            return self.val

    def __pos__(self):
        if False:
            while True:
                i = 10
        return self

    def __neg__(self):
        if False:
            while True:
                i = 10
        return self.__class__(-self.val)

    @classmethod
    def _get_val(cls, other):
        if False:
            print('Hello World!')
        if isinstance(other, cls):
            return other.val
        else:
            try:
                return cls.dom.convert(other)
            except CoercionFailed:
                return None

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        val = self._get_val(other)
        if val is not None:
            return self.__class__(self.val + val)
        else:
            return NotImplemented

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.__add__(other)

    def __sub__(self, other):
        if False:
            print('Hello World!')
        val = self._get_val(other)
        if val is not None:
            return self.__class__(self.val - val)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if False:
            print('Hello World!')
        return (-self).__add__(other)

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        val = self._get_val(other)
        if val is not None:
            return self.__class__(self.val * val)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.__mul__(other)

    def __truediv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        val = self._get_val(other)
        if val is not None:
            return self.__class__(self.val * self._invert(val))
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if False:
            print('Hello World!')
        return self.invert().__mul__(other)

    def __mod__(self, other):
        if False:
            while True:
                i = 10
        val = self._get_val(other)
        if val is not None:
            return self.__class__(self.val % val)
        else:
            return NotImplemented

    def __rmod__(self, other):
        if False:
            while True:
                i = 10
        val = self._get_val(other)
        if val is not None:
            return self.__class__(val % self.val)
        else:
            return NotImplemented

    def __pow__(self, exp):
        if False:
            return 10
        if not exp:
            return self.__class__(self.dom.one)
        if exp < 0:
            (val, exp) = (self.invert().val, -exp)
        else:
            val = self.val
        return self.__class__(pow(val, int(exp), self.mod))

    def _compare(self, other, op):
        if False:
            while True:
                i = 10
        val = self._get_val(other)
        if val is None:
            return NotImplemented
        return op(self.val, val % self.mod)

    def _compare_deprecated(self, other, op):
        if False:
            i = 10
            return i + 15
        val = self._get_val(other)
        if val is None:
            return NotImplemented
        sympy_deprecation_warning('Ordered comparisons with modular integers are deprecated.\n\n            Use e.g. int(a) < int(b) instead of a < b.\n            ', deprecated_since_version='1.13', active_deprecations_target='modularinteger-compare', stacklevel=4)
        return op(self.val, val % self.mod)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self._compare(other, operator.eq)

    def __ne__(self, other):
        if False:
            return 10
        return self._compare(other, operator.ne)

    def __lt__(self, other):
        if False:
            return 10
        return self._compare_deprecated(other, operator.lt)

    def __le__(self, other):
        if False:
            i = 10
            return i + 15
        return self._compare_deprecated(other, operator.le)

    def __gt__(self, other):
        if False:
            return 10
        return self._compare_deprecated(other, operator.gt)

    def __ge__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._compare_deprecated(other, operator.ge)

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.val)

    @classmethod
    def _invert(cls, value):
        if False:
            while True:
                i = 10
        return cls.dom.invert(value, cls.mod)

    def invert(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__(self._invert(self.val))
_modular_integer_cache: dict[tuple[Any, Any, Any], type[ModularInteger]] = {}

def ModularIntegerFactory(_mod, _dom, _sym, parent):
    if False:
        print('Hello World!')
    'Create custom class for specific integer modulus.'
    try:
        _mod = _dom.convert(_mod)
    except CoercionFailed:
        ok = False
    else:
        ok = True
    if not ok or _mod < 1:
        raise ValueError('modulus must be a positive integer, got %s' % _mod)
    key = (_mod, _dom, _sym)
    try:
        cls = _modular_integer_cache[key]
    except KeyError:

        class cls(ModularInteger):
            (mod, dom, sym) = (_mod, _dom, _sym)
            _parent = parent
        if _sym:
            cls.__name__ = 'SymmetricModularIntegerMod%s' % _mod
        else:
            cls.__name__ = 'ModularIntegerMod%s' % _mod
        _modular_integer_cache[key] = cls
    return cls