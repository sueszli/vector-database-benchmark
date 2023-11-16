def test_block_number(get_contract_with_gas_estimation, w3):
    if False:
        print('Hello World!')
    block_number_code = '\n@external\ndef block_number() -> uint256:\n    return block.number\n    '
    c = get_contract_with_gas_estimation(block_number_code)
    assert c.block_number() == 1
    w3.testing.mine(1)
    assert c.block_number() == 2