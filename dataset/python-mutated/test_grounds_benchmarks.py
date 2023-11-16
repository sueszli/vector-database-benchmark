from __future__ import annotations
import pytest
from ibis.common.annotations import attribute
from ibis.common.collections import frozendict
from ibis.common.grounds import Annotable, Concrete
pytestmark = pytest.mark.benchmark

class MyObject(Concrete):
    a: int
    b: str
    c: tuple[int, ...]
    d: frozendict[str, int]

    @attribute
    def e(self):
        if False:
            i = 10
            return i + 15
        return self.a * 2

    @attribute
    def f(self):
        if False:
            i = 10
            return i + 15
        return self.b * 2

    @attribute
    def g(self):
        if False:
            while True:
                i = 10
        return self.c * 2

def test_concrete_construction(benchmark):
    if False:
        print('Hello World!')
    benchmark(MyObject, 1, '2', c=(3, 4), d=frozendict(e=5, f=6))

def test_concrete_isinstance(benchmark):
    if False:
        while True:
            i = 10

    def check(obj):
        if False:
            i = 10
            return i + 15
        for _ in range(100):
            assert isinstance(obj, MyObject)
            assert isinstance(obj, Concrete)
            assert isinstance(obj, Annotable)
    obj = MyObject(1, '2', c=(3, 4), d=frozendict(e=5, f=6))
    benchmark(check, obj)