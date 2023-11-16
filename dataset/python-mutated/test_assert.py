import pytest
from eth_tester.exceptions import TransactionFailed

def _fixup_err_str(s):
    if False:
        print('Hello World!')
    return s.replace('execution reverted: ', '')

def test_assert_refund(w3, get_contract_with_gas_estimation, assert_tx_failed):
    if False:
        return 10
    code = '\n@external\ndef foo():\n    raise\n    '
    c = get_contract_with_gas_estimation(code)
    a0 = w3.eth.accounts[0]
    gas_sent = 10 ** 6
    tx_hash = c.foo(transact={'from': a0, 'gas': gas_sent, 'gasPrice': 10})
    tx_receipt = w3.eth.get_transaction_receipt(tx_hash)
    assert tx_receipt['status'] == 0
    assert tx_receipt['gasUsed'] < gas_sent

def test_assert_reason(w3, get_contract_with_gas_estimation, assert_tx_failed, memory_mocker):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef test(a: int128) -> int128:\n    assert a > 1, "larger than one please"\n    return 1 + a\n\n@external\ndef test2(a: int128, b: int128, extra_reason: String[32]) -> int128:\n    c: int128 = 11\n    assert a > 1, "a is not large enough"\n    assert b == 1, concat("b may only be 1", extra_reason)\n    return a + b + c\n\n@external\ndef test3(reason_str: String[32]):\n    raise reason_str\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.test(2) == 3
    with pytest.raises(TransactionFailed) as e_info:
        c.test(0)
    assert _fixup_err_str(e_info.value.args[0]) == 'larger than one please'
    with pytest.raises(TransactionFailed) as e_info:
        c.test2(0, 1, '')
    assert _fixup_err_str(e_info.value.args[0]) == 'a is not large enough'
    with pytest.raises(TransactionFailed) as e_info:
        c.test2(2, 2, ' because I said so')
    assert _fixup_err_str(e_info.value.args[0]) == 'b may only be 1' + ' because I said so'
    assert c.test2(5, 1, '') == 17
    with pytest.raises(TransactionFailed) as e_info:
        c.test3('An exception')
    assert _fixup_err_str(e_info.value.args[0]) == 'An exception'
invalid_code = ['\n@external\ndef test(a: int128) -> int128:\n    assert a > 1, ""\n    return 1 + a\n    ', '\n@external\ndef test(a: int128) -> int128:\n    raise ""\n    ', '\n@external\ndef test():\n    assert create_minimal_proxy_to(self)\n    ']

@pytest.mark.parametrize('code', invalid_code)
def test_invalid_assertions(get_contract, assert_compile_failed, code):
    if False:
        print('Hello World!')
    assert_compile_failed(lambda : get_contract(code))
valid_code = ['\n@external\ndef mint(_to: address, _value: uint256):\n    raise\n    ', '\n@internal\ndef ret1() -> int128:\n    return 1\n@external\ndef test():\n    assert self.ret1() == 1\n    ', '\n@internal\ndef valid_address(sender: address) -> bool:\n    selfdestruct(sender)\n@external\ndef test():\n    assert self.valid_address(msg.sender)\n    ', "\n@external\ndef test():\n    assert raw_call(msg.sender, b'', max_outsize=1, gas=10, value=1000*1000) == b''\n    ", '\n@external\ndef test():\n    assert create_minimal_proxy_to(self) == self\n    ']

@pytest.mark.parametrize('code', valid_code)
def test_valid_assertions(get_contract, code):
    if False:
        i = 10
        return i + 15
    get_contract(code)

def test_assert_staticcall(get_contract, assert_tx_failed, memory_mocker):
    if False:
        return 10
    foreign_code = '\nstate: uint256\n@external\ndef not_really_constant() -> uint256:\n    self.state += 1\n    return self.state\n    '
    code = '\ninterface ForeignContract:\n    def not_really_constant() -> uint256: view\n\n@external\ndef test():\n    assert ForeignContract(msg.sender).not_really_constant() == 1\n    '
    c1 = get_contract(foreign_code)
    c2 = get_contract(code, *[c1.address])
    assert_tx_failed(lambda : c2.test())

def test_assert_in_for_loop(get_contract, assert_tx_failed, memory_mocker):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef test(x: uint256[3]) -> bool:\n    for i in range(3):\n        assert x[i] < 5\n    return True\n    '
    c = get_contract(code)
    c.test([1, 2, 3])
    assert_tx_failed(lambda : c.test([5, 1, 3]))
    assert_tx_failed(lambda : c.test([1, 5, 3]))
    assert_tx_failed(lambda : c.test([1, 3, 5]))

def test_assert_with_reason_in_for_loop(get_contract, assert_tx_failed, memory_mocker):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef test(x: uint256[3]) -> bool:\n    for i in range(3):\n        assert x[i] < 5, "because reasons"\n    return True\n    '
    c = get_contract(code)
    c.test([1, 2, 3])
    assert_tx_failed(lambda : c.test([5, 1, 3]))
    assert_tx_failed(lambda : c.test([1, 5, 3]))
    assert_tx_failed(lambda : c.test([1, 3, 5]))

def test_assert_reason_revert_length(w3, get_contract, assert_tx_failed, memory_mocker):
    if False:
        return 10
    code = '\n@external\ndef test() -> int128:\n    assert 1 == 2, "oops"\n    return 1\n'
    c = get_contract(code)
    assert_tx_failed(lambda : c.test(), exc_text='oops')