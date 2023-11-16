from decimal import Decimal
import pytest
from vyper.semantics.types import IntegerT

def test_minmax(get_contract_with_gas_estimation):
    if False:
        return 10
    minmax_test = '\n@external\ndef foo() -> decimal:\n    return min(3.0, 5.0) + max(10.0, 20.0) + min(200.1, 400.0) + max(3000.0, 8000.02) + min(50000.003, 70000.004)  # noqa: E501\n\n@external\ndef goo() -> uint256:\n    return min(3, 5) + max(40, 80)\n    '
    c = get_contract_with_gas_estimation(minmax_test)
    assert c.foo() == Decimal('58223.123')
    assert c.goo() == 83
    print('Passed min/max test')

@pytest.mark.parametrize('return_type', sorted(IntegerT.all()))
def test_minmax_var_and_literal_and_bultin(get_contract_with_gas_estimation, return_type):
    if False:
        return 10
    '\n    Tests to verify that min and max work as expected when a variable/literal\n    and a literal are passed for all integer types.\n    '
    (lo, hi) = return_type.ast_bounds
    code = f'\n@external\ndef foo() -> {return_type}:\n    a: {return_type} = {hi}\n    b: {return_type} = 5\n    return max(a, 5)\n\n@external\ndef bar() -> {return_type}:\n    a: {return_type} = {lo}\n    b: {return_type} = 5\n    return min(a, 5)\n\n@external\ndef both_literals_max() -> {return_type}:\n    return max({hi}, 2)\n\n@external\ndef both_literals_min() -> {return_type}:\n    return min({lo}, 2)\n\n@external\ndef both_builtins_max() -> {return_type}:\n    return max(min_value({return_type}), max_value({return_type}))\n\n@external\ndef both_builtins_min() -> {return_type}:\n    return min(min_value({return_type}), max_value({return_type}))\n'
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == hi
    assert c.bar() == lo
    assert c.both_literals_max() == hi
    assert c.both_literals_min() == lo
    assert c.both_builtins_max() == hi
    assert c.both_builtins_min() == lo

def test_max_var_uint256_literal_int128(get_contract_with_gas_estimation):
    if False:
        return 10
    '\n    Tests to verify that max works as expected when a variable/literal uint256\n    and a literal int128 are passed.\n    '
    code = '\n@external\ndef foo() -> uint256:\n    a: uint256 = 2 ** 200\n    b: uint256 = 5\n    return max(a, 5) + max(b, 5)\n\n@external\ndef goo() -> uint256:\n    a: uint256 = 2 ** 200\n    b: uint256 = 5\n    return max(5, a) + max(5, b)\n\n@external\ndef bar() -> uint256:\n    a: uint256 = 2\n    b: uint256 = 5\n    return max(a, 5) + max(b, 5)\n\n@external\ndef baz() -> uint256:\n    a: uint256 = 2\n    b: uint256 = 5\n    return max(5, a) + max(5, b)\n\n@external\ndef both_literals() -> uint256:\n    return max(2 ** 200, 2)\n'
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == 2 ** 200 + 5
    assert c.goo() == 2 ** 200 + 5
    assert c.bar() == 5 + 5
    assert c.baz() == 5 + 5
    assert c.both_literals() == 2 ** 200

def test_min_var_uint256_literal_int128(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests to verify that max works as expected when a variable/literal uint256\n    and a literal int128 are passed.\n    '
    code = '\n@external\ndef foo() -> uint256:\n    a: uint256 = 2 ** 200\n    b: uint256 = 5\n    return min(a, 5) + min(b, 5)\n\n@external\ndef goo() -> uint256:\n    a: uint256 = 2 ** 200\n    b: uint256 = 5\n    return min(5, a) + min(5, b)\n\n@external\ndef bar() -> uint256:\n    a: uint256 = 2\n    b: uint256 = 5\n    return min(a, 5) + min(b, 5)\n\n@external\ndef baz() -> uint256:\n    a: uint256 = 2\n    b: uint256 = 5\n    return min(5, a) + min(5, b)\n\n@external\ndef both_literals() -> uint256:\n    return min(2 ** 200, 2)\n'
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == 5 + 5
    assert c.goo() == 5 + 5
    assert c.bar() == 2 + 5
    assert c.baz() == 2 + 5
    assert c.both_literals() == 2

def test_minmax_var_uint256_var_int128(get_contract_with_gas_estimation, assert_compile_failed):
    if False:
        i = 10
        return i + 15
    '\n    Tests to verify that max throws an error if a variable uint256 and a\n    variable int128 are passed.\n    '
    from vyper.exceptions import TypeMismatch
    code_1 = '\n@external\ndef foo() -> uint256:\n    a: uint256 = 2\n    b: int128 = 3\n    return max(a, b)\n'
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code_1), TypeMismatch)
    code_2 = '\n@external\ndef foo() -> uint256:\n    a: uint256 = 2\n    b: int128 = 3\n    return max(b, a)\n'
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code_2), TypeMismatch)
    code_3 = '\n@external\ndef foo() -> uint256:\n    a: uint256 = 2\n    b: int128 = 3\n    return min(a, b)\n'
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code_3), TypeMismatch)
    code_4 = '\n@external\ndef foo() -> uint256:\n    a: uint256 = 2\n    b: int128 = 3\n    return min(b, a)\n'
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code_4), TypeMismatch)

def test_minmax_var_uint256_negative_int128(get_contract_with_gas_estimation, assert_tx_failed, assert_compile_failed):
    if False:
        print('Hello World!')
    from vyper.exceptions import TypeMismatch
    code_1 = '\n@external\ndef foo() -> uint256:\n    a: uint256 = 2 ** 200\n    return max(a, -1)\n'
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code_1), TypeMismatch)
    code_2 = '\n@external\ndef foo() -> uint256:\n    a: uint256 = 2 ** 200\n    return min(a, -1)\n'
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code_2), TypeMismatch)

def test_unsigned(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\n@external\ndef foo1() -> uint256:\n    return min(0, 2**255)\n\n@external\ndef foo2() -> uint256:\n    return min(2**255, 0)\n\n@external\ndef foo3() -> uint256:\n    return max(0, 2**255)\n\n@external\ndef foo4() -> uint256:\n    return max(2**255, 0)\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo1() == 0
    assert c.foo2() == 0
    assert c.foo3() == 2 ** 255
    assert c.foo4() == 2 ** 255

def test_signed(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef foo1() -> int128:\n    return min(min_value(int128), max_value(int128))\n\n@external\ndef foo2() -> int128:\n    return min(max_value(int128), min_value(int128))\n\n@external\ndef foo3() -> int128:\n    return max(min_value(int128), max_value(int128))\n\n@external\ndef foo4() -> int128:\n    return max(max_value(int128), min_value(int128))\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo1() == -2 ** 127
    assert c.foo2() == -2 ** 127
    assert c.foo3() == 2 ** 127 - 1
    assert c.foo4() == 2 ** 127 - 1