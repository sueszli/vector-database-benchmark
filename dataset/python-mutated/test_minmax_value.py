import pytest
from vyper.exceptions import InvalidType, OverflowException
from vyper.semantics.types import DecimalT, IntegerT
from vyper.semantics.types.shortcuts import INT256_T, UINT256_T

@pytest.mark.parametrize('typ', sorted(IntegerT.all() + (DecimalT(),)))
@pytest.mark.parametrize('op', ('min_value', 'max_value'))
def test_minmax_value(get_contract, op, typ):
    if False:
        while True:
            i = 10
    code = f'\n@external\ndef foo() -> {typ}:\n    return {op}({typ})\n    '
    c = get_contract(code)
    (lo, hi) = typ.ast_bounds
    if op == 'min_value':
        assert c.foo() == lo
    elif op == 'max_value':
        assert c.foo() == hi

@pytest.mark.parametrize('typ', sorted(IntegerT.all()))
def test_minmax_value_int_oob(get_contract, assert_compile_failed, typ):
    if False:
        return 10
    upper = f'\n@external\ndef foo():\n    a: {typ} = max_value({typ}) + 1\n    '
    lower = f'\n@external\ndef foo():\n    a: {typ} = min_value({typ}) - 1\n    '
    if typ == UINT256_T:
        assert_compile_failed(lambda : get_contract(upper), OverflowException)
    else:
        assert_compile_failed(lambda : get_contract(upper), InvalidType)
    if typ == INT256_T:
        assert_compile_failed(lambda : get_contract(lower), OverflowException)
    else:
        assert_compile_failed(lambda : get_contract(lower), InvalidType)

@pytest.mark.parametrize('typ', [DecimalT()])
def test_minmax_value_decimal_oob(get_contract, assert_compile_failed, typ):
    if False:
        while True:
            i = 10
    upper = f'\n@external\ndef foo():\n    a: {typ} = max_value({typ}) + 1e-10\n    '
    lower = f'\n@external\ndef foo():\n    a: {typ} = min_value({typ}) - 1e-10\n    '
    assert_compile_failed(lambda : get_contract(upper), OverflowException)
    assert_compile_failed(lambda : get_contract(lower), OverflowException)