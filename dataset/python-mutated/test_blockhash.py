def test_blockhash(get_contract_with_gas_estimation, w3):
    if False:
        return 10
    w3.testing.mine(1)
    block_number_code = '\n@external\ndef prev() -> bytes32:\n    return block.prevhash\n\n@external\ndef previous_blockhash() -> bytes32:\n    return blockhash(block.number - 1)\n'
    c = get_contract_with_gas_estimation(block_number_code)
    assert c.prev() == c.previous_blockhash()

def test_negative_blockhash(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\n@external\ndef foo() -> bytes32:\n    return blockhash(-1)\n'
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code))

def test_too_old_blockhash(assert_tx_failed, get_contract_with_gas_estimation, w3):
    if False:
        for i in range(10):
            print('nop')
    w3.testing.mine(257)
    code = '\n@external\ndef get_50_blockhash() -> bytes32:\n    return blockhash(block.number - 257)\n'
    c = get_contract_with_gas_estimation(code)
    assert_tx_failed(lambda : c.get_50_blockhash())

def test_non_existing_blockhash(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef get_future_blockhash() -> bytes32:\n    return blockhash(block.number + 1)\n'
    c = get_contract_with_gas_estimation(code)
    assert_tx_failed(lambda : c.get_future_blockhash())