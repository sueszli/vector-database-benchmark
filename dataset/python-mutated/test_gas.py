def test_gas_call(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    gas_call = '\n@external\ndef foo() -> uint256:\n    return msg.gas\n    '
    c = get_contract_with_gas_estimation(gas_call)
    assert c.foo(call={'gas': 50000}) < 50000
    assert c.foo(call={'gas': 50000}) > 25000