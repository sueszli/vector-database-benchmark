import pytest
from vyper.exceptions import StructureException

def test_continue1(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\n@external\ndef foo() -> bool:\n    for i in range(2):\n        continue\n        return False\n    return True\n'
    c = get_contract_with_gas_estimation(code)
    assert c.foo()

def test_continue2(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef foo() -> int128:\n    x: int128 = 0\n    for i in range(3):\n        x += 1\n        continue\n        x -= 1\n    return x\n'
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == 3

def test_continue3(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo() -> int128:\n    x: int128 = 0\n    for i in range(3):\n        x += i\n        continue\n    return x\n'
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == 3

def test_continue4(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo() -> int128:\n    x: int128 = 0\n    for i in range(6):\n        if i % 2 == 0:\n            continue\n        x += 1\n    return x\n'
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == 3
fail_list = [('\n@external\ndef foo():\n    a: uint256 = 3\n    continue\n    ', StructureException), ('\n@external\ndef foo():\n    if True:\n        continue\n    ', StructureException), ('\n@external\ndef foo():\n    for i in [1, 2, 3]:\n        b: uint256 = i\n    if True:\n        continue\n    ', StructureException)]

@pytest.mark.parametrize('bad_code,exc', fail_list)
def test_block_fail(assert_compile_failed, get_contract_with_gas_estimation, bad_code, exc):
    if False:
        for i in range(10):
            print('nop')
    assert_compile_failed(lambda : get_contract_with_gas_estimation(bad_code), exc)