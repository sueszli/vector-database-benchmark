"""
Reversed Operations not available in the stdlib operator module.
Defining these instead of using lambdas allows us to reference them by name.
"""
from __future__ import annotations
import operator

def radd(left, right):
    if False:
        while True:
            i = 10
    return right + left

def rsub(left, right):
    if False:
        while True:
            i = 10
    return right - left

def rmul(left, right):
    if False:
        i = 10
        return i + 15
    return right * left

def rdiv(left, right):
    if False:
        return 10
    return right / left

def rtruediv(left, right):
    if False:
        i = 10
        return i + 15
    return right / left

def rfloordiv(left, right):
    if False:
        while True:
            i = 10
    return right // left

def rmod(left, right):
    if False:
        return 10
    if isinstance(right, str):
        typ = type(left).__name__
        raise TypeError(f'{typ} cannot perform the operation mod')
    return right % left

def rdivmod(left, right):
    if False:
        return 10
    return divmod(right, left)

def rpow(left, right):
    if False:
        for i in range(10):
            print('nop')
    return right ** left

def rand_(left, right):
    if False:
        return 10
    return operator.and_(right, left)

def ror_(left, right):
    if False:
        for i in range(10):
            print('nop')
    return operator.or_(right, left)

def rxor(left, right):
    if False:
        i = 10
        return i + 15
    return operator.xor(right, left)