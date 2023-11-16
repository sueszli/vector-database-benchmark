import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from vyper import ast as vy_ast
from vyper.builtins import functions as vy_fn
from vyper.exceptions import OverflowException

@pytest.mark.fuzzing
@settings(max_examples=50)
@given(a=st.integers(min_value=-2 ** 255 + 1, max_value=2 ** 255 - 1))
@example(a=0)
def test_abs(get_contract, a):
    if False:
        for i in range(10):
            print('nop')
    source = '\n@external\ndef foo(a: int256) -> int256:\n    return abs(a)\n    '
    contract = get_contract(source)
    vyper_ast = vy_ast.parse_to_ast(f'abs({a})')
    old_node = vyper_ast.body[0].value
    new_node = vy_fn.DISPATCH_TABLE['abs'].evaluate(old_node)
    assert contract.foo(a) == new_node.value == abs(a)

@pytest.mark.fuzzing
@settings(max_examples=50)
@given(a=st.integers(min_value=2 ** 255, max_value=2 ** 256 - 1))
def test_abs_upper_bound_folding(get_contract, a):
    if False:
        while True:
            i = 10
    source = f'\n@external\ndef foo(a: int256) -> int256:\n    return abs({a})\n    '
    with pytest.raises(OverflowException):
        get_contract(source)

def test_abs_lower_bound(get_contract, assert_tx_failed):
    if False:
        return 10
    source = '\n@external\ndef foo(a: int256) -> int256:\n    return abs(a)\n    '
    contract = get_contract(source)
    assert_tx_failed(lambda : contract.foo(-2 ** 255))

def test_abs_lower_bound_folded(get_contract, assert_tx_failed):
    if False:
        i = 10
        return i + 15
    source = '\n@external\ndef foo() -> int256:\n    return abs(-2**255)\n    '
    with pytest.raises(OverflowException):
        get_contract(source)