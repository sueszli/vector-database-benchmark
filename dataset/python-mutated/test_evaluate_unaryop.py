import pytest
from vyper import ast as vy_ast

@pytest.mark.parametrize('bool_cond', [True, False])
def test_unaryop(get_contract, bool_cond):
    if False:
        i = 10
        return i + 15
    source = '\n@external\ndef foo(a: bool) -> bool:\n    return not a\n    '
    contract = get_contract(source)
    vyper_ast = vy_ast.parse_to_ast(f'not {bool_cond}')
    old_node = vyper_ast.body[0].value
    new_node = old_node.evaluate()
    assert contract.foo(bool_cond) == new_node.value

@pytest.mark.parametrize('count', range(2, 11))
@pytest.mark.parametrize('bool_cond', [True, False])
def test_unaryop_nested(get_contract, bool_cond, count):
    if False:
        while True:
            i = 10
    source = f"\n@external\ndef foo(a: bool) -> bool:\n    return {'not ' * count} a\n    "
    contract = get_contract(source)
    literal_op = f"{'not ' * count}{bool_cond}"
    vyper_ast = vy_ast.parse_to_ast(literal_op)
    vy_ast.folding.replace_literal_ops(vyper_ast)
    expected = vyper_ast.body[0].value.value
    assert contract.foo(bool_cond) == expected