def test_arbitration_code(w3, get_contract_with_gas_estimation, assert_tx_failed):
    if False:
        i = 10
        return i + 15
    arbitration_code = '\nbuyer: address\nseller: address\narbitrator: address\n\n@external\ndef setup(_seller: address, _arbitrator: address):\n    if self.buyer == empty(address):\n        self.buyer = msg.sender\n        self.seller = _seller\n        self.arbitrator = _arbitrator\n\n@external\ndef finalize():\n    assert msg.sender == self.buyer or msg.sender == self.arbitrator\n    send(self.seller, self.balance)\n\n@external\ndef refund():\n    assert msg.sender == self.seller or msg.sender == self.arbitrator\n    send(self.buyer, self.balance)\n\n    '
    (a0, a1, a2) = w3.eth.accounts[:3]
    c = get_contract_with_gas_estimation(arbitration_code, value=1)
    c.setup(a1, a2, transact={})
    assert_tx_failed(lambda : c.finalize(transact={'from': a1}))
    c.finalize(transact={})
    print('Passed escrow test')

def test_arbitration_code_with_init(w3, assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    arbitration_code_with_init = '\nbuyer: address\nseller: address\narbitrator: address\n\n@external\n@payable\ndef __init__(_seller: address, _arbitrator: address):\n    if self.buyer == empty(address):\n        self.buyer = msg.sender\n        self.seller = _seller\n        self.arbitrator = _arbitrator\n\n@external\ndef finalize():\n    assert msg.sender == self.buyer or msg.sender == self.arbitrator\n    send(self.seller, self.balance)\n\n@external\ndef refund():\n    assert msg.sender == self.seller or msg.sender == self.arbitrator\n    send(self.buyer, self.balance)\n    '
    (a0, a1, a2) = w3.eth.accounts[:3]
    c = get_contract_with_gas_estimation(arbitration_code_with_init, *[a1, a2], value=1)
    assert_tx_failed(lambda : c.finalize(transact={'from': a1}))
    c.finalize(transact={'from': a0})
    print('Passed escrow test with initializer')