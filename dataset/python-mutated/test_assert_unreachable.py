def test_unreachable_refund(w3, get_contract):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo():\n    assert msg.sender != msg.sender, UNREACHABLE\n    '
    c = get_contract(code)
    a0 = w3.eth.accounts[0]
    gas_sent = 10 ** 6
    tx_hash = c.foo(transact={'from': a0, 'gas': gas_sent, 'gasPrice': 10})
    tx_receipt = w3.eth.get_transaction_receipt(tx_hash)
    assert tx_receipt['status'] == 0
    assert tx_receipt['gasUsed'] == gas_sent

def test_basic_unreachable(w3, get_contract, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef foo(val: int128) -> bool:\n    assert val > 0, UNREACHABLE\n    assert val == 2, UNREACHABLE\n    return True\n    '
    c = get_contract(code)
    assert c.foo(2) is True
    assert_tx_failed(lambda : c.foo(1), exc_text='Invalid opcode 0xfe')
    assert_tx_failed(lambda : c.foo(-1), exc_text='Invalid opcode 0xfe')
    assert_tx_failed(lambda : c.foo(-2), exc_text='Invalid opcode 0xfe')

def test_basic_call_unreachable(w3, get_contract, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    code = '\n\n@view\n@internal\ndef _test_me(val: int128) -> bool:\n    return val == 33\n\n@external\ndef foo(val: int128) -> int128:\n    assert self._test_me(val), UNREACHABLE\n    return -123\n    '
    c = get_contract(code)
    assert c.foo(33) == -123
    assert_tx_failed(lambda : c.foo(1), exc_text='Invalid opcode 0xfe')
    assert_tx_failed(lambda : c.foo(-1), exc_text='Invalid opcode 0xfe')

def test_raise_unreachable(w3, get_contract, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef foo():\n    raise UNREACHABLE\n    '
    c = get_contract(code)
    assert_tx_failed(lambda : c.foo(), exc_text='Invalid opcode 0xfe')