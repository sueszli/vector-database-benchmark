import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from vyper import ast as vy_ast
from vyper.exceptions import UnfoldableNode

@pytest.mark.fuzzing
@settings(max_examples=50)
@given(left=st.integers(), right=st.integers())
@pytest.mark.parametrize('op', ['==', '!=', '<', '<=', '>=', '>'])
def test_compare_eq_signed(get_contract, op, left, right):
    if False:
        print('Hello World!')
    source = f'\n@external\ndef foo(a: int128, b: int128) -> bool:\n    return a {op} b\n    '
    contract = get_contract(source)
    vyper_ast = vy_ast.parse_to_ast(f'{left} {op} {right}')
    old_node = vyper_ast.body[0].value
    new_node = old_node.evaluate()
    assert contract.foo(left, right) == new_node.value

@pytest.mark.fuzzing
@settings(max_examples=50)
@given(left=st.integers(min_value=0), right=st.integers(min_value=0))
@pytest.mark.parametrize('op', ['==', '!=', '<', '<=', '>=', '>'])
def test_compare_eq_unsigned(get_contract, op, left, right):
    if False:
        i = 10
        return i + 15
    source = f'\n@external\ndef foo(a: uint128, b: uint128) -> bool:\n    return a {op} b\n    '
    contract = get_contract(source)
    vyper_ast = vy_ast.parse_to_ast(f'{left} {op} {right}')
    old_node = vyper_ast.body[0].value
    new_node = old_node.evaluate()
    assert contract.foo(left, right) == new_node.value

@pytest.mark.fuzzing
@settings(max_examples=20)
@given(left=st.integers(), right=st.lists(st.integers(), min_size=1, max_size=16))
def test_compare_in(left, right, get_contract):
    if False:
        print('Hello World!')
    source = f'\n@external\ndef foo(a: int128, b: int128[{len(right)}]) -> bool:\n    c: int128[{len(right)}] = b\n    return a in c\n\n@external\ndef bar(a: int128) -> bool:\n    # note: codegen unrolls to `a == right[0] or a == right[1] ...`\n    return a in {right}\n    '
    contract = get_contract(source)
    vyper_ast = vy_ast.parse_to_ast(f'{left} in {right}')
    old_node = vyper_ast.body[0].value
    new_node = old_node.evaluate()
    assert contract.foo(left, right) == new_node.value
    assert contract.bar(left) == new_node.value
    assert (left in right) == new_node.value

@pytest.mark.fuzzing
@settings(max_examples=20)
@given(left=st.integers(), right=st.lists(st.integers(), min_size=1, max_size=16))
def test_compare_not_in(left, right, get_contract):
    if False:
        print('Hello World!')
    source = f'\n@external\ndef foo(a: int128, b: int128[{len(right)}]) -> bool:\n    c: int128[{len(right)}] = b\n    return a not in c\n\n@external\ndef bar(a: int128) -> bool:\n    # note: codegen unrolls to `a != right[0] and a != right[1] ...`\n    return a not in {right}\n    '
    contract = get_contract(source)
    vyper_ast = vy_ast.parse_to_ast(f'{left} not in {right}')
    old_node = vyper_ast.body[0].value
    new_node = old_node.evaluate()
    assert contract.foo(left, right) == new_node.value
    assert contract.bar(left) == new_node.value
    assert (left not in right) == new_node.value

@pytest.mark.parametrize('op', ['==', '!=', '<', '<=', '>=', '>'])
def test_compare_type_mismatch(op):
    if False:
        return 10
    vyper_ast = vy_ast.parse_to_ast(f'1 {op} 1.0')
    old_node = vyper_ast.body[0].value
    with pytest.raises(UnfoldableNode):
        old_node.evaluate()