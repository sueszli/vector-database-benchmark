def test_throw_on_sending(w3, assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\nx: public(int128)\n\n@external\ndef __init__():\n    self.x = 123\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.x() == 123
    assert w3.eth.get_balance(c.address) == 0
    assert_tx_failed(lambda : w3.eth.send_transaction({'to': c.address, 'value': w3.to_wei(0.1, 'ether')}))
    assert w3.eth.get_balance(c.address) == 0

def test_basic_default(w3, get_logs, get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\nevent Sent:\n    sender: indexed(address)\n\n@external\n@payable\ndef __default__():\n    log Sent(msg.sender)\n    '
    c = get_contract_with_gas_estimation(code)
    logs = get_logs(w3.eth.send_transaction({'to': c.address, 'value': 10 ** 17}), c, 'Sent')
    assert w3.eth.accounts[0] == logs[0].args.sender
    assert w3.eth.get_balance(c.address) == w3.to_wei(0.1, 'ether')

def test_basic_default_default_param_function(w3, get_logs, get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\nevent Sent:\n    sender: indexed(address)\n\n@external\n@payable\ndef fooBar(a: int128 = 12345) -> int128:\n    log Sent(empty(address))\n    return a\n\n@external\n@payable\ndef __default__():\n    log Sent(msg.sender)\n    '
    c = get_contract_with_gas_estimation(code)
    logs = get_logs(w3.eth.send_transaction({'to': c.address, 'value': 10 ** 17}), c, 'Sent')
    assert w3.eth.accounts[0] == logs[0].args.sender
    assert w3.eth.get_balance(c.address) == w3.to_wei(0.1, 'ether')

def test_basic_default_not_payable(w3, assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\nevent Sent:\n    sender: indexed(address)\n\n@external\ndef __default__():\n    log Sent(msg.sender)\n    '
    c = get_contract_with_gas_estimation(code)
    assert_tx_failed(lambda : w3.eth.send_transaction({'to': c.address, 'value': 10 ** 17}))

def test_multi_arg_default(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\n@payable\n@external\ndef __default__(arg1: int128):\n    pass\n    '
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code))

def test_always_public(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\n@internal\ndef __default__():\n    pass\n    '
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code))

def test_always_public_2(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\nevent Sent:\n    sender: indexed(address)\n\ndef __default__():\n    log Sent(msg.sender)\n    '
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code))

def test_zero_method_id(w3, get_logs, get_contract, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    code = '\nevent Sent:\n    sig: uint256\n\n@external\n@payable\n# function selector: 0x00000000\ndef blockHashAskewLimitary(v: uint256) -> uint256:\n    log Sent(2)\n    return 7\n\n@external\ndef __default__():\n    log Sent(1)\n    '
    c = get_contract(code)
    assert c.blockHashAskewLimitary(0) == 7

    def _call_with_bytes(hexstr):
        if False:
            for i in range(10):
                print('nop')
        logs = get_logs(w3.eth.send_transaction({'to': c.address, 'value': 0, 'data': hexstr}), c, 'Sent')
        return logs[0].args.sig
    assert 1 == _call_with_bytes('0x')
    assert 2 == _call_with_bytes('0x' + '00' * 36)
    assert 2 == _call_with_bytes('0x' + '00' * 37)
    for i in range(4):
        assert 1 == _call_with_bytes('0x' + '00' * i)
    for i in range(4, 36):
        assert_tx_failed(lambda : _call_with_bytes('0x' + '00' * i))

def test_another_zero_method_id(w3, get_logs, get_contract, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    code = '\nevent Sent:\n    sig: uint256\n\n@external\n@payable\n# function selector: 0x00000000\ndef wycpnbqcyf() -> uint256:\n    log Sent(2)\n    return 7\n\n@external\ndef __default__():\n    log Sent(1)\n    '
    c = get_contract(code)
    assert c.wycpnbqcyf() == 7

    def _call_with_bytes(hexstr):
        if False:
            i = 10
            return i + 15
        logs = get_logs(w3.eth.send_transaction({'to': c.address, 'value': 0, 'data': hexstr}), c, 'Sent')
        return logs[0].args.sig
    assert 1 == _call_with_bytes('0x')
    assert 2 == _call_with_bytes('0x' + '00' * 4)
    assert 2 == _call_with_bytes('0x' + '00' * 5)
    for i in range(4):
        assert 1 == _call_with_bytes('0x' + '00' * i)

def test_partial_selector_match_trailing_zeroes(w3, get_logs, get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\nevent Sent:\n    sig: uint256\n\n@external\n@payable\n# function selector: 0xd88e0b00\ndef fow() -> uint256:\n    log Sent(2)\n    return 7\n\n@external\ndef __default__():\n    log Sent(1)\n    '
    c = get_contract(code)
    assert c.fow() == 7

    def _call_with_bytes(hexstr):
        if False:
            while True:
                i = 10
        logs = get_logs(w3.eth.send_transaction({'to': c.address, 'value': 0, 'data': hexstr}), c, 'Sent')
        return logs[0].args.sig
    assert 1 == _call_with_bytes('0x')
    assert 2 == _call_with_bytes('0xd88e0b00')
    assert 1 == _call_with_bytes('0xd88e0b')