def test_crowdfund(w3, tester, get_contract_with_gas_estimation_for_constants):
    if False:
        while True:
            i = 10
    crowdfund = '\n\nstruct Funder:\n    sender: address\n    value: uint256\nfunders: HashMap[int128, Funder]\nnextFunderIndex: int128\nbeneficiary: address\ndeadline: public(uint256)\ngoal: public(uint256)\nrefundIndex: int128\ntimelimit: public(uint256)\n\n@external\ndef __init__(_beneficiary: address, _goal: uint256, _timelimit: uint256):\n    self.beneficiary = _beneficiary\n    self.deadline = block.timestamp + _timelimit\n    self.timelimit = _timelimit\n    self.goal = _goal\n\n@external\n@payable\ndef participate():\n    assert block.timestamp < self.deadline\n    nfi: int128 = self.nextFunderIndex\n    self.funders[nfi].sender = msg.sender\n    self.funders[nfi].value = msg.value\n    self.nextFunderIndex = nfi + 1\n\n@external\n@view\ndef expired() -> bool:\n    return block.timestamp >= self.deadline\n\n@external\n@view\ndef block_timestamp() -> uint256:\n    return block.timestamp\n\n@external\n@view\ndef reached() -> bool:\n    return self.balance >= self.goal\n\n@external\ndef finalize():\n    assert block.timestamp >= self.deadline and self.balance >= self.goal\n    selfdestruct(self.beneficiary)\n\n@external\ndef refund():\n    ind: int128 = self.refundIndex\n    for i in range(ind, ind + 30):\n        if i >= self.nextFunderIndex:\n            self.refundIndex = self.nextFunderIndex\n            return\n        send(self.funders[i].sender, self.funders[i].value)\n        self.funders[i].sender = 0x0000000000000000000000000000000000000000\n        self.funders[i].value = 0\n    self.refundIndex = ind + 30\n\n    '
    (a0, a1, a2, a3, a4, a5, a6) = w3.eth.accounts[:7]
    c = get_contract_with_gas_estimation_for_constants(crowdfund, *[a1, 50, 60])
    c.participate(transact={'value': 5})
    assert c.timelimit() == 60
    assert c.deadline() - c.block_timestamp() == 59
    assert not c.expired()
    assert not c.reached()
    c.participate(transact={'value': 49})
    assert c.reached()
    pre_bal = w3.eth.get_balance(a1)
    w3.testing.mine(100)
    assert c.expired()
    c.finalize(transact={})
    post_bal = w3.eth.get_balance(a1)
    assert post_bal - pre_bal == 54
    c = get_contract_with_gas_estimation_for_constants(crowdfund, *[a1, 50, 60])
    c.participate(transact={'value': 1, 'from': a3})
    c.participate(transact={'value': 2, 'from': a4})
    c.participate(transact={'value': 3, 'from': a5})
    c.participate(transact={'value': 4, 'from': a6})
    w3.testing.mine(100)
    assert c.expired()
    assert not c.reached()
    pre_bals = [w3.eth.get_balance(x) for x in [a3, a4, a5, a6]]
    c.refund(transact={})
    post_bals = [w3.eth.get_balance(x) for x in [a3, a4, a5, a6]]
    assert [y - x for (x, y) in zip(pre_bals, post_bals)] == [1, 2, 3, 4]

def test_crowdfund2(w3, tester, get_contract_with_gas_estimation_for_constants):
    if False:
        while True:
            i = 10
    crowdfund2 = '\nstruct Funder:\n    sender: address\n    value: uint256\n\nfunders: HashMap[int128, Funder]\nnextFunderIndex: int128\nbeneficiary: address\ndeadline: public(uint256)\ngoal: uint256\nrefundIndex: int128\ntimelimit: public(uint256)\n\n@external\ndef __init__(_beneficiary: address, _goal: uint256, _timelimit: uint256):\n    self.beneficiary = _beneficiary\n    self.deadline = block.timestamp + _timelimit\n    self.timelimit = _timelimit\n    self.goal = _goal\n\n@external\n@payable\ndef participate():\n    assert block.timestamp < self.deadline\n    nfi: int128 = self.nextFunderIndex\n    self.funders[nfi] = Funder({sender: msg.sender, value: msg.value})\n    self.nextFunderIndex = nfi + 1\n\n@external\n@view\ndef expired() -> bool:\n    return block.timestamp >= self.deadline\n\n@external\n@view\ndef block_timestamp() -> uint256:\n    return block.timestamp\n\n@external\n@view\ndef reached() -> bool:\n    return self.balance >= self.goal\n\n@external\ndef finalize():\n    assert block.timestamp >= self.deadline and self.balance >= self.goal\n    selfdestruct(self.beneficiary)\n\n@external\ndef refund():\n    ind: int128 = self.refundIndex\n    for i in range(ind, ind + 30):\n        if i >= self.nextFunderIndex:\n            self.refundIndex = self.nextFunderIndex\n            return\n        send(self.funders[i].sender, self.funders[i].value)\n        self.funders[i] = empty(Funder)\n    self.refundIndex = ind + 30\n\n    '
    (a0, a1, a2, a3, a4, a5, a6) = w3.eth.accounts[:7]
    c = get_contract_with_gas_estimation_for_constants(crowdfund2, *[a1, 50, 60])
    c.participate(transact={'value': 5})
    assert c.timelimit() == 60
    assert c.deadline() - c.block_timestamp() == 59
    assert not c.expired()
    assert not c.reached()
    c.participate(transact={'value': 49})
    assert c.reached()
    pre_bal = w3.eth.get_balance(a1)
    w3.testing.mine(100)
    assert c.expired()
    c.finalize(transact={})
    post_bal = w3.eth.get_balance(a1)
    assert post_bal - pre_bal == 54
    c = get_contract_with_gas_estimation_for_constants(crowdfund2, *[a1, 50, 60])
    c.participate(transact={'value': 1, 'from': a3})
    c.participate(transact={'value': 2, 'from': a4})
    c.participate(transact={'value': 3, 'from': a5})
    c.participate(transact={'value': 4, 'from': a6})
    w3.testing.mine(100)
    assert c.expired()
    assert not c.reached()
    pre_bals = [w3.eth.get_balance(x) for x in [a3, a4, a5, a6]]
    c.refund(transact={})
    post_bals = [w3.eth.get_balance(x) for x in [a3, a4, a5, a6]]
    assert [y - x for (x, y) in zip(pre_bals, post_bals)] == [1, 2, 3, 4]