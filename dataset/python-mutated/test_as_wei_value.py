from decimal import Decimal
import pytest
wei_denoms = {'femtoether': 3, 'kwei': 3, 'babbage': 3, 'picoether': 6, 'mwei': 6, 'lovelace': 6, 'nanoether': 9, 'gwei': 9, 'shannon': 9, 'microether': 12, 'szabo': 12, 'milliether': 15, 'finney': 15, 'ether': 18, 'kether': 21, 'grand': 21}

@pytest.mark.parametrize('denom,multiplier', wei_denoms.items())
def test_wei_uint256(get_contract, assert_tx_failed, denom, multiplier):
    if False:
        print('Hello World!')
    code = f'\n@external\ndef foo(a: uint256) -> uint256:\n    return as_wei_value(a, "{denom}")\n    '
    c = get_contract(code)
    value = (2 ** 256 - 1) // 10 ** multiplier
    assert c.foo(value) == value * 10 ** multiplier
    value = (2 ** 256 - 1) // 10 ** (multiplier - 1)
    assert_tx_failed(lambda : c.foo(value))

@pytest.mark.parametrize('denom,multiplier', wei_denoms.items())
def test_wei_int128(get_contract, assert_tx_failed, denom, multiplier):
    if False:
        return 10
    code = f'\n@external\ndef foo(a: int128) -> uint256:\n    return as_wei_value(a, "{denom}")\n    '
    c = get_contract(code)
    value = (2 ** 127 - 1) // 10 ** multiplier
    assert c.foo(value) == value * 10 ** multiplier

@pytest.mark.parametrize('denom,multiplier', wei_denoms.items())
def test_wei_decimal(get_contract, assert_tx_failed, denom, multiplier):
    if False:
        for i in range(10):
            print('nop')
    code = f'\n@external\ndef foo(a: decimal) -> uint256:\n    return as_wei_value(a, "{denom}")\n    '
    c = get_contract(code)
    value = Decimal((2 ** 127 - 1) / 10 ** multiplier)
    assert c.foo(value) == value * 10 ** multiplier

@pytest.mark.parametrize('value', (-1, -2 ** 127))
@pytest.mark.parametrize('data_type', ['decimal', 'int128'])
def test_negative_value_reverts(get_contract, assert_tx_failed, value, data_type):
    if False:
        return 10
    code = f'\n@external\ndef foo(a: {data_type}) -> uint256:\n    return as_wei_value(a, "ether")\n    '
    c = get_contract(code)
    assert_tx_failed(lambda : c.foo(value))

@pytest.mark.parametrize('denom,multiplier', wei_denoms.items())
@pytest.mark.parametrize('data_type', ['decimal', 'int128', 'uint256'])
def test_zero_value(get_contract, assert_tx_failed, denom, multiplier, data_type):
    if False:
        print('Hello World!')
    code = f'\n@external\ndef foo(a: {data_type}) -> uint256:\n    return as_wei_value(a, "{denom}")\n    '
    c = get_contract(code)
    assert c.foo(0) == 0

def test_ext_call(w3, side_effects_contract, assert_side_effects_invoked, get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef foo(a: Foo) -> uint256:\n    return as_wei_value(a.foo(7), "ether")\n\ninterface Foo:\n    def foo(x: uint8) -> uint8: nonpayable\n    '
    c1 = side_effects_contract('uint8')
    c2 = get_contract(code)
    assert c2.foo(c1.address) == w3.to_wei(7, 'ether')
    assert_side_effects_invoked(c1, lambda : c2.foo(c1.address, transact={}))

def test_internal_call(w3, get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\n@external\ndef foo() -> uint256:\n    return as_wei_value(self.bar(), "ether")\n\n@internal\ndef bar() -> uint8:\n    return 7\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == w3.to_wei(7, 'ether')