from vyper.exceptions import TypeMismatch

def test_basic_in_list(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef testin(x: int128) -> bool:\n    y: int128 = 1\n    s: int128[4]  = [1, 2, 3, 4]\n    if (x + 1) in s:\n        return True\n    return False\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.testin(0) is True
    assert c.testin(1) is True
    assert c.testin(2) is True
    assert c.testin(3) is True
    assert c.testin(4) is False
    assert c.testin(5) is False
    assert c.testin(-1) is False

def test_in_storage_list(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\nallowed: int128[10]\n\n@external\ndef in_test(x: int128) -> bool:\n    self.allowed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n    if x in self.allowed:\n        return True\n    return False\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.in_test(1) is True
    assert c.in_test(9) is True
    assert c.in_test(11) is False
    assert c.in_test(-1) is False
    assert c.in_test(32000) is False

def test_in_calldata_list(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef in_test(x: int128, y: int128[10]) -> bool:\n    if x in y:\n        return True\n    return False\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.in_test(1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) is True
    assert c.in_test(9, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) is True
    assert c.in_test(11, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) is False
    assert c.in_test(-1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) is False
    assert c.in_test(32000, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) is False

def test_cmp_in_list(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\n@external\ndef in_test(x: int128) -> bool:\n    if x in [9, 7, 6, 5]:\n        return True\n    return False\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.in_test(1) is False
    assert c.in_test(-7) is False
    assert c.in_test(-9) is False
    assert c.in_test(5) is True
    assert c.in_test(7) is True

def test_cmp_not_in_list(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef in_test(x: int128) -> bool:\n    if x not in [9, 7, 6, 5]:\n        return True\n    return False\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.in_test(1) is True
    assert c.in_test(-7) is True
    assert c.in_test(-9) is True
    assert c.in_test(5) is False
    assert c.in_test(7) is False

def test_mixed_in_list(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\n@external\ndef testin() -> bool:\n    s: int128[4] = [1, 2, 3, 4]\n    if "test" in s:\n        return True\n    return False\n    '
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code), TypeMismatch)

def test_ownership(w3, assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n\nowners: address[2]\n\n@external\ndef __init__():\n    self.owners[0] = msg.sender\n\n@external\ndef set_owner(i: int128, new_owner: address):\n    assert msg.sender in self.owners\n    self.owners[i] = new_owner\n\n@external\ndef is_owner() -> bool:\n    return msg.sender in self.owners\n    '
    a1 = w3.eth.accounts[1]
    c = get_contract_with_gas_estimation(code)
    assert c.is_owner() is True
    assert c.is_owner(call={'from': a1}) is False
    assert_tx_failed(lambda : c.set_owner(1, a1, call={'from': a1}))
    c.set_owner(1, a1, transact={})
    assert c.is_owner(call={'from': a1}) is True
    c.set_owner(0, a1, transact={})
    assert c.is_owner() is False

def test_in_fails_when_types_dont_match(get_contract_with_gas_estimation, assert_tx_failed):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef testin(x: address) -> bool:\n    s: int128[4] = [1, 2, 3, 4]\n    if x in s:\n        return True\n    return False\n'
    assert_tx_failed(lambda : get_contract_with_gas_estimation(code), TypeMismatch)