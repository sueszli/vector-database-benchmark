def test_tx_gasprice(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef tx_gasprice() -> uint256:\n    return tx.gasprice\n'
    c = get_contract(code)
    for i in range(10):
        assert c.tx_gasprice(call={'gasPrice': 10 ** i}) == 10 ** i