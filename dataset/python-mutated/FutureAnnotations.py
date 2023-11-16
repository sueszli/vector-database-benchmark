from __future__ import annotations
from collections.abc import Mapping
from numbers import Integral
from typing import List

def concrete_types(a: int, b: bool, c: list):
    if False:
        while True:
            i = 10
    assert a == 42, repr(a)
    assert b is False, repr(b)
    assert c == [1, 'kaksi'], repr(c)

def abcs(a: Integral, b: Mapping):
    if False:
        return 10
    assert a == 42, repr(a)
    assert b == {'key': 'value'}, repr(b)

def typing_(a: List, b: List[int]):
    if False:
        return 10
    assert a == ['foo', 'bar'], repr(a)
    assert b == [1, 2, 3], repr(b)

def invalid1(a: foo):
    if False:
        return 10
    assert a == 'xxx'

def invalid2(a: 1 / 0):
    if False:
        for i in range(10):
            print('nop')
    assert a == 'xxx'