from decimal import Decimal
import pytest
from vyper.exceptions import ZeroDivisionException

def test_modulo(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\n@external\ndef num_modulo_num() -> int128:\n    return 1 % 2\n\n@external\ndef decimal_modulo_decimal() -> decimal:\n    return 1.5 % .33\n\n@external\ndef decimal_modulo_num() -> decimal:\n    return .5 % 1.0\n\n\n@external\ndef num_modulo_decimal() -> decimal:\n    return 1.5 % 1.0\n'
    c = get_contract_with_gas_estimation(code)
    assert c.num_modulo_num() == 1
    assert c.decimal_modulo_decimal() == Decimal('.18')
    assert c.decimal_modulo_num() == Decimal('.5')
    assert c.num_modulo_decimal() == Decimal('.5')

def test_modulo_with_input_of_zero(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef foo(a: decimal, b: decimal) -> decimal:\n    return a % b\n'
    c = get_contract_with_gas_estimation(code)
    assert_tx_failed(lambda : c.foo(Decimal('1'), Decimal('0')))

def test_literals_vs_evm(get_contract):
    if False:
        print('Hello World!')
    code = '\n@external\n@view\ndef foo() -> (int128, int128, int128, int128):\n    return 5%2, 5%-2, -5%2, -5%-2\n\n@external\n@view\ndef bar(a: int128) -> bool:\n    assert -5%2 == a%2\n    return True\n'
    c = get_contract(code)
    assert c.foo() == [1, 1, -1, -1]
    assert c.bar(-5) is True
BAD_CODE = ['\n@external\ndef foo() -> uint256:\n    return 2 % 0\n    ', '\n@external\ndef foo() -> int128:\n    return -2 % 0\n    ', '\n@external\ndef foo() -> decimal:\n    return 2.22 % 0.0\n    ', '\n@external\ndef foo(a: uint256) -> uint256:\n    return a % 0\n    ', '\n@external\ndef foo(a: int128) -> int128:\n    return a % 0\n    ', '\n@external\ndef foo(a: decimal) -> decimal:\n    return a % 0.0\n    ']

@pytest.mark.parametrize('code', BAD_CODE)
def test_modulo_by_zero(code, assert_compile_failed, get_contract):
    if False:
        for i in range(10):
            print('nop')
    assert_compile_failed(lambda : get_contract(code), ZeroDivisionException)