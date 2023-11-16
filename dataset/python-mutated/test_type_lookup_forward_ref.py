from typing import Dict, ForwardRef, List, Union
import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import ResolutionFailed
from tests.common import utils

@given(st.data())
def test_mutually_recursive_types_with_typevar(data):
    if False:
        while True:
            i = 10
    A = Dict[bool, 'B']
    B = Union[List[bool], A]
    with pytest.raises(ResolutionFailed, match="Could not resolve ForwardRef\\('B'\\)"):
        data.draw(st.from_type(A))
    with utils.temp_registered(ForwardRef('B'), lambda _: st.deferred(lambda : b_strategy)):
        b_strategy = st.from_type(B)
        data.draw(b_strategy)
        data.draw(st.from_type(A))
        data.draw(st.from_type(B))

@given(st.data())
def test_mutually_recursive_types_with_typevar_alternate(data):
    if False:
        for i in range(10):
            print('nop')
    C = Union[List[bool], 'D']
    D = Dict[bool, C]
    with pytest.raises(ResolutionFailed, match="Could not resolve ForwardRef\\('D'\\)"):
        data.draw(st.from_type(C))
    with utils.temp_registered(ForwardRef('D'), lambda _: st.deferred(lambda : d_strategy)):
        d_strategy = st.from_type(D)
        data.draw(d_strategy)
        data.draw(st.from_type(C))
        data.draw(st.from_type(D))