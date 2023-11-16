import pytest
from eth_tester.exceptions import TransactionFailed
from vyper import compiler
from vyper.exceptions import StructureException, TypeMismatch

def test_variable_assignment(get_contract, keccak):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo() -> Bytes[4]:\n    bar: Bytes[4] = slice(msg.data, 0, 4)\n    return bar\n'
    contract = get_contract(code)
    assert contract.foo() == bytes(keccak(text='foo()')[:4])

def test_slicing_start_index_other_than_zero(get_contract):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(_value: uint256) -> uint256:\n    bar: Bytes[32] = slice(msg.data, 4, 32)\n    return convert(bar, uint256)\n'
    contract = get_contract(code)
    assert contract.foo(42) == 42

def test_get_full_calldata(get_contract, keccak, w3):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef foo(bar: uint256) -> Bytes[36]:\n    data: Bytes[36] = slice(msg.data, 0, 36)\n    return data\n'
    contract = get_contract(code)
    method_id = keccak(text='foo(uint256)').hex()[2:10]
    encoded_42 = w3.to_bytes(42).hex()
    expected_result = method_id + '00' * 31 + encoded_42
    assert contract.foo(42).hex() == expected_result

@pytest.mark.parametrize('bar', [0, 1, 42, 2 ** 256 - 1])
def test_calldata_private(get_contract, bar):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(bar: uint256) -> uint256:\n    data: Bytes[32] = slice(msg.data, 4, 32)\n    return convert(data, uint256)\n    '
    c = get_contract(code)
    assert c.foo(bar) == bar

def test_memory_pointer_advances_appropriately(get_contract, keccak):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef foo() -> (uint256, Bytes[4], uint256):\n    a: uint256 = max_value(uint256)\n    b: Bytes[4] = slice(msg.data, 0, 4)\n    c: uint256 = max_value(uint256)\n\n    return (a, b, c)\n'
    contract = get_contract(code)
    assert contract.foo() == [2 ** 256 - 1, bytes(keccak(text='foo()')[:4]), 2 ** 256 - 1]

def test_assignment_to_storage(w3, get_contract, keccak):
    if False:
        while True:
            i = 10
    code = '\ncache: public(Bytes[4])\n\n@external\ndef foo():\n    self.cache = slice(msg.data, 0, 4)\n'
    acct = w3.eth.accounts[0]
    contract = get_contract(code)
    contract.foo(transact={'from': acct})
    assert contract.cache() == bytes(keccak(text='foo()')[:4])

def test_get_len(get_contract):
    if False:
        return 10
    code = '\n@external\ndef foo(bar: uint256) -> uint256:\n    return len(msg.data)\n'
    contract = get_contract(code)
    assert contract.foo(42) == 36
fail_list = [('\n@external\ndef foo() -> Bytes[4]:\n    bar: Bytes[4] = msg.data\n    return bar\n    ', StructureException), ('\n@external\ndef foo() -> Bytes[7]:\n    bar: Bytes[7] = concat(msg.data, 0xc0ffee)\n    return bar\n    ', StructureException), ('\n@external\ndef foo() -> uint256:\n    bar: uint256 = convert(msg.data, uint256)\n    return bar\n    ', StructureException), ('\na: HashMap[Bytes[10], uint256]\n\n@external\ndef foo():\n    self.a[msg.data] += 1\n    ', StructureException), ('\n@external\ndef foo(bar: uint256) -> bytes32:\n    ret_val: bytes32 = slice(msg.data, 4, 32)\n    return ret_val\n    ', TypeMismatch)]

@pytest.mark.parametrize('bad_code', fail_list)
def test_invalid_usages_compile_error(bad_code):
    if False:
        while True:
            i = 10
    with pytest.raises(bad_code[1]):
        compiler.compile_code(bad_code[0])

def test_runtime_failure_bounds_check(get_contract):
    if False:
        return 10
    code = '\n@external\ndef foo(_value: uint256) -> uint256:\n    val: Bytes[40] = slice(msg.data, 0, 40)\n    return convert(slice(val, 4, 32), uint256)\n'
    contract = get_contract(code)
    with pytest.raises(TransactionFailed):
        contract.foo(42)