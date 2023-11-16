from __future__ import annotations
from typing import TypedDict, Union
import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import InvalidArgument
alias = Union[int, str]

class A(TypedDict):
    a: int

class B(TypedDict):
    a: A
    b: alias

@given(st.from_type(B))
def test_complex_forward_ref_in_typed_dict(d):
    if False:
        while True:
            i = 10
    assert isinstance(d['a'], dict)
    assert isinstance(d['a']['a'], int)
    assert isinstance(d['b'], (int, str))

def test_complex_forward_ref_in_typed_dict_local():
    if False:
        return 10
    local_alias = Union[int, str]

    class C(TypedDict):
        a: A
        b: local_alias
    c_strategy = st.from_type(C)
    with pytest.raises(InvalidArgument):
        c_strategy.example()