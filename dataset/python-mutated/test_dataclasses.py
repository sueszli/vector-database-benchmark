from dataclasses import dataclass, field, InitVar
from hypothesis import given, settings, strategies as st
from torch.testing._internal.jit_utils import JitTestCase
from typing import List, Optional
import sys
import torch
import unittest
from enum import Enum

@dataclass(order=True)
class Point:
    x: float
    y: float
    norm: Optional[torch.Tensor] = None

    def __post_init__(self):
        if False:
            while True:
                i = 10
        self.norm = (torch.tensor(self.x) ** 2 + torch.tensor(self.y) ** 2) ** 0.5

class MixupScheme(Enum):
    INPUT = ['input']
    MANIFOLD = ['input', 'before_fusion_projection', 'after_fusion_projection', 'after_classifier_projection']

@dataclass
class MixupParams:

    def __init__(self, alpha: float=0.125, scheme: MixupScheme=MixupScheme.INPUT):
        if False:
            for i in range(10):
                print('nop')
        self.alpha = alpha
        self.scheme = scheme

class MixupScheme2(Enum):
    A = 1
    B = 2

@dataclass
class MixupParams2:

    def __init__(self, alpha: float=0.125, scheme: MixupScheme2=MixupScheme2.A):
        if False:
            print('Hello World!')
        self.alpha = alpha
        self.scheme = scheme

@dataclass
class MixupParams3:

    def __init__(self, alpha: float=0.125, scheme: MixupScheme2=MixupScheme2.A):
        if False:
            print('Hello World!')
        self.alpha = alpha
        self.scheme = scheme
NonHugeFloats = st.floats(min_value=-10000.0, max_value=10000.0, allow_nan=False)

class TestDataclasses(JitTestCase):

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        torch._C._jit_clear_class_registry()

    def test_init_vars(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        @dataclass(order=True)
        class Point2:
            x: float
            y: float
            norm_p: InitVar[int] = 2
            norm: Optional[torch.Tensor] = None

            def __post_init__(self, norm_p: int):
                if False:
                    while True:
                        i = 10
                self.norm = (torch.tensor(self.x) ** norm_p + torch.tensor(self.y) ** norm_p) ** (1 / norm_p)

        def fn(x: float, y: float, p: int):
            if False:
                print('Hello World!')
            pt = Point2(x, y, p)
            return pt.norm
        self.checkScript(fn, (1.0, 2.0, 3))

    @settings(deadline=None)
    @given(NonHugeFloats, NonHugeFloats)
    def test__post_init__(self, x, y):
        if False:
            return 10
        P = torch.jit.script(Point)

        def fn(x: float, y: float):
            if False:
                for i in range(10):
                    print('nop')
            pt = P(x, y)
            return pt.norm
        self.checkScript(fn, [x, y])

    @settings(deadline=None)
    @given(st.tuples(NonHugeFloats, NonHugeFloats), st.tuples(NonHugeFloats, NonHugeFloats))
    def test_comparators(self, pt1, pt2):
        if False:
            return 10
        (x1, y1) = pt1
        (x2, y2) = pt2
        P = torch.jit.script(Point)

        def compare(x1: float, y1: float, x2: float, y2: float):
            if False:
                print('Hello World!')
            pt1 = P(x1, y1)
            pt2 = P(x2, y2)
            return (pt1 == pt2, pt1 < pt2, pt1 <= pt2, pt1 > pt2, pt1 >= pt2)
        self.checkScript(compare, [x1, y1, x2, y2])

    def test_default_factories(self):
        if False:
            return 10

        @dataclass
        class Foo(object):
            x: List[int] = field(default_factory=list)
        with self.assertRaises(NotImplementedError):
            torch.jit.script(Foo)

            def fn():
                if False:
                    while True:
                        i = 10
                foo = Foo()
                return foo.x
            torch.jit.script(fn)()

    def test_custom__eq__(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        @dataclass
        class CustomEq:
            a: int
            b: int

            def __eq__(self, other: 'CustomEq') -> bool:
                if False:
                    i = 10
                    return i + 15
                return self.a == other.a

        def fn(a: int, b1: int, b2: int):
            if False:
                print('Hello World!')
            pt1 = CustomEq(a, b1)
            pt2 = CustomEq(a, b2)
            return pt1 == pt2
        self.checkScript(fn, [1, 2, 3])

    def test_no_source(self):
        if False:
            return 10
        with self.assertRaises(RuntimeError):
            torch.jit.script(MixupParams)
        torch.jit.script(MixupParams2)

    def test_use_unregistered_dataclass_raises(self):
        if False:
            while True:
                i = 10

        def f(a: MixupParams3):
            if False:
                i = 10
                return i + 15
            return 0
        with self.assertRaises(OSError):
            torch.jit.script(f)