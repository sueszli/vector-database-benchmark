"""Logical boolean operators: not, and, or."""
from nvidia.dali._autograph.utils import hooks

def not_(a):
    if False:
        return 10
    'Functional form of "not".'
    if hooks._DISPATCH.detect_overload_not_(a):
        return hooks._DISPATCH.not_(a)
    return _py_not(a)

def _py_not(a):
    if False:
        for i in range(10):
            print('nop')
    'Default Python implementation of the "not_" operator.'
    return not a

def and_(a, b):
    if False:
        print('Hello World!')
    'Functional form of "and". Uses lazy evaluation semantics.'
    a_val = a()
    if hooks._DISPATCH.detect_overload_lazy_and(a_val):
        return hooks._DISPATCH.lazy_and(a_val, b)
    return _py_lazy_and(a_val, b)

def _py_lazy_and(cond, b):
    if False:
        while True:
            i = 10
    'Lazy-eval equivalent of "and" in Python.'
    return cond and b()

def or_(a, b):
    if False:
        return 10
    'Functional form of "or". Uses lazy evaluation semantics.'
    a_val = a()
    if hooks._DISPATCH.detect_overload_lazy_or(a_val):
        return hooks._DISPATCH.lazy_or(a_val, b)
    return _py_lazy_or(a_val, b)

def _py_lazy_or(cond, b):
    if False:
        print('Hello World!')
    'Lazy-eval equivalent of "or" in Python.'
    return cond or b()

def eq(a, b):
    if False:
        for i in range(10):
            print('nop')
    'Functional form of "equal".'
    if hooks._DISPATCH.detect_overload_equal(a) or hooks._DISPATCH.detect_overload_equal(b):
        return hooks._DISPATCH.equal(a, b)
    return _py_equal(a, b)

def _py_equal(a, b):
    if False:
        while True:
            i = 10
    'Overload of "equal" that falls back to Python\'s default implementation.'
    return a == b

def not_eq(a, b):
    if False:
        i = 10
        return i + 15
    'Functional form of "not-equal".'
    return not_(eq(a, b))