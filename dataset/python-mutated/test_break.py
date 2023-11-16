from decimal import Decimal
import pytest
from vyper.exceptions import StructureException

def test_break_test(get_contract_with_gas_estimation):
    if False:
        return 10
    break_test = '\n@external\ndef foo(n: decimal) -> int128:\n    c: decimal = n * 1.0\n    output: int128 = 0\n    for i in range(400):\n        c = c / 1.2589\n        if c < 1.0:\n            output = i\n            break\n    return output\n    '
    c = get_contract_with_gas_estimation(break_test)
    assert c.foo(Decimal('1')) == 0
    assert c.foo(Decimal('2')) == 3
    assert c.foo(Decimal('10')) == 10
    assert c.foo(Decimal('200')) == 23
    print('Passed for-loop break test')

def test_break_test_2(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    break_test_2 = '\n@external\ndef foo(n: decimal) -> int128:\n    c: decimal = n * 1.0\n    output: int128 = 0\n    for i in range(40):\n        if c < 10.0:\n            output = i * 10\n            break\n        c = c / 10.0\n    for i in range(10):\n        c = c / 1.2589\n        if c < 1.0:\n            output = output + i\n            break\n    return output\n    '
    c = get_contract_with_gas_estimation(break_test_2)
    assert c.foo(Decimal('1')) == 0
    assert c.foo(Decimal('2')) == 3
    assert c.foo(Decimal('10')) == 10
    assert c.foo(Decimal('200')) == 23
    assert c.foo(Decimal('4000000')) == 66
    print('Passed for-loop break test 2')

def test_break_test_3(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    break_test_3 = '\n@external\ndef foo(n: int128) -> int128:\n    c: decimal = convert(n, decimal)\n    output: int128 = 0\n    for i in range(40):\n        if c < 10.0:\n            output = i * 10\n            break\n        c /= 10.0\n    for i in range(10):\n        c /= 1.2589\n        if c < 1.0:\n            output = output + i\n            break\n    return output\n    '
    c = get_contract_with_gas_estimation(break_test_3)
    assert c.foo(1) == 0
    assert c.foo(2) == 3
    assert c.foo(10) == 10
    assert c.foo(200) == 23
    assert c.foo(4000000) == 66
    print('Passed aug-assignment break composite test')
fail_list = [('\n@external\ndef foo():\n    a: uint256 = 3\n    break\n    ', StructureException), ('\n@external\ndef foo():\n    if True:\n        break\n    ', StructureException), ('\n@external\ndef foo():\n    for i in [1, 2, 3]:\n        b: uint256 = i\n    if True:\n        break\n    ', StructureException)]

@pytest.mark.parametrize('bad_code,exc', fail_list)
def test_block_fail(assert_compile_failed, get_contract_with_gas_estimation, bad_code, exc):
    if False:
        for i in range(10):
            print('nop')
    assert_compile_failed(lambda : get_contract_with_gas_estimation(bad_code), exc)