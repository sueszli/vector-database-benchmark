import functools
import types
from ._make import _make_ne
_operation_names = {'eq': '==', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}

def cmp_using(eq=None, lt=None, le=None, gt=None, ge=None, require_same_type=True, class_name='Comparable'):
    if False:
        while True:
            i = 10
    "\n    Create a class that can be passed into `attrs.field`'s ``eq``, ``order``,\n    and ``cmp`` arguments to customize field comparison.\n\n    The resulting class will have a full set of ordering methods if at least\n    one of ``{lt, le, gt, ge}`` and ``eq``  are provided.\n\n    :param Optional[callable] eq: `callable` used to evaluate equality of two\n        objects.\n    :param Optional[callable] lt: `callable` used to evaluate whether one\n        object is less than another object.\n    :param Optional[callable] le: `callable` used to evaluate whether one\n        object is less than or equal to another object.\n    :param Optional[callable] gt: `callable` used to evaluate whether one\n        object is greater than another object.\n    :param Optional[callable] ge: `callable` used to evaluate whether one\n        object is greater than or equal to another object.\n\n    :param bool require_same_type: When `True`, equality and ordering methods\n        will return `NotImplemented` if objects are not of the same type.\n\n    :param Optional[str] class_name: Name of class. Defaults to 'Comparable'.\n\n    See `comparison` for more details.\n\n    .. versionadded:: 21.1.0\n    "
    body = {'__slots__': ['value'], '__init__': _make_init(), '_requirements': [], '_is_comparable_to': _is_comparable_to}
    num_order_functions = 0
    has_eq_function = False
    if eq is not None:
        has_eq_function = True
        body['__eq__'] = _make_operator('eq', eq)
        body['__ne__'] = _make_ne()
    if lt is not None:
        num_order_functions += 1
        body['__lt__'] = _make_operator('lt', lt)
    if le is not None:
        num_order_functions += 1
        body['__le__'] = _make_operator('le', le)
    if gt is not None:
        num_order_functions += 1
        body['__gt__'] = _make_operator('gt', gt)
    if ge is not None:
        num_order_functions += 1
        body['__ge__'] = _make_operator('ge', ge)
    type_ = types.new_class(class_name, (object,), {}, lambda ns: ns.update(body))
    if require_same_type:
        type_._requirements.append(_check_same_type)
    if 0 < num_order_functions < 4:
        if not has_eq_function:
            msg = 'eq must be define is order to complete ordering from lt, le, gt, ge.'
            raise ValueError(msg)
        type_ = functools.total_ordering(type_)
    return type_

def _make_init():
    if False:
        print('Hello World!')
    '\n    Create __init__ method.\n    '

    def __init__(self, value):
        if False:
            while True:
                i = 10
        '\n        Initialize object with *value*.\n        '
        self.value = value
    return __init__

def _make_operator(name, func):
    if False:
        return 10
    '\n    Create operator method.\n    '

    def method(self, other):
        if False:
            i = 10
            return i + 15
        if not self._is_comparable_to(other):
            return NotImplemented
        result = func(self.value, other.value)
        if result is NotImplemented:
            return NotImplemented
        return result
    method.__name__ = f'__{name}__'
    method.__doc__ = f'Return a {_operation_names[name]} b.  Computed by attrs.'
    return method

def _is_comparable_to(self, other):
    if False:
        while True:
            i = 10
    '\n    Check whether `other` is comparable to `self`.\n    '
    return all((func(self, other) for func in self._requirements))

def _check_same_type(self, other):
    if False:
        return 10
    '\n    Return True if *self* and *other* are of the same type, False otherwise.\n    '
    return other.value.__class__ is self.value.__class__