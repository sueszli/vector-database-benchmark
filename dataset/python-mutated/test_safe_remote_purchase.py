import pytest

@pytest.fixture
def contract_code(get_contract):
    if False:
        while True:
            i = 10
    with open('examples/safe_remote_purchase/safe_remote_purchase.vy') as f:
        contract_code = f.read()
    return contract_code

@pytest.fixture
def get_balance(w3):
    if False:
        while True:
            i = 10

    def get_balance():
        if False:
            print('Hello World!')
        (a0, a1) = w3.eth.accounts[:2]
        return (w3.eth.get_balance(a0), w3.eth.get_balance(a1))
    return get_balance

def test_initial_state(w3, assert_tx_failed, get_contract, get_balance, contract_code):
    if False:
        while True:
            i = 10
    assert_tx_failed(lambda : get_contract(contract_code, value=13))
    (a0_pre_bal, a1_pre_bal) = get_balance()
    c = get_contract(contract_code, value_in_eth=2)
    assert c.seller() == w3.eth.accounts[0]
    assert c.value() == w3.to_wei(1, 'ether')
    assert c.unlocked() is True
    assert get_balance() == (a0_pre_bal - w3.to_wei(2, 'ether'), a1_pre_bal)

def test_abort(w3, assert_tx_failed, get_balance, get_contract, contract_code):
    if False:
        i = 10
        return i + 15
    (a0, a1, a2) = w3.eth.accounts[:3]
    (a0_pre_bal, a1_pre_bal) = get_balance()
    c = get_contract(contract_code, value=w3.to_wei(2, 'ether'))
    assert c.value() == w3.to_wei(1, 'ether')
    assert_tx_failed(lambda : c.abort(transact={'from': a2}))
    c.abort(transact={'from': a0})
    assert get_balance() == (a0_pre_bal, a1_pre_bal)
    c = get_contract(contract_code, value=2)
    c.purchase(transact={'value': 2, 'from': a1})
    assert_tx_failed(lambda : c.abort(transact={'from': a0}))

def test_purchase(w3, get_contract, assert_tx_failed, get_balance, contract_code):
    if False:
        i = 10
        return i + 15
    (a0, a1, a2, a3) = w3.eth.accounts[:4]
    (init_bal_a0, init_bal_a1) = get_balance()
    c = get_contract(contract_code, value=2)
    assert_tx_failed(lambda : c.purchase(transact={'value': 1, 'from': a1}))
    assert_tx_failed(lambda : c.purchase(transact={'value': 3, 'from': a1}))
    c.purchase(transact={'value': 2, 'from': a1})
    assert c.buyer() == a1
    assert c.unlocked() is False
    assert get_balance() == (init_bal_a0 - 2, init_bal_a1 - 2)
    assert_tx_failed(lambda : c.purchase(transact={'value': 2, 'from': a3}))

def test_received(w3, get_contract, assert_tx_failed, get_balance, contract_code):
    if False:
        i = 10
        return i + 15
    (a0, a1) = w3.eth.accounts[:2]
    (init_bal_a0, init_bal_a1) = get_balance()
    c = get_contract(contract_code, value=2)
    assert_tx_failed(lambda : c.received(transact={'from': a1}))
    c.purchase(transact={'value': 2, 'from': a1})
    assert_tx_failed(lambda : c.received(transact={'from': a0}))
    c.received(transact={'from': a1})
    assert get_balance() == (init_bal_a0 + 1, init_bal_a1 - 1)

def test_received_reentrancy(w3, get_contract, assert_tx_failed, get_balance, contract_code):
    if False:
        while True:
            i = 10
    buyer_contract_code = '\ninterface PurchaseContract:\n\n    def received(): nonpayable\n    def purchase(): payable\n    def unlocked() -> bool: view\n\npurchase_contract: PurchaseContract\n\n\n@external\ndef __init__(_purchase_contract: address):\n    self.purchase_contract = PurchaseContract(_purchase_contract)\n\n\n@payable\n@external\ndef start_purchase():\n    self.purchase_contract.purchase(value=2)\n\n\n@payable\n@external\ndef start_received():\n    self.purchase_contract.received()\n\n\n@external\n@payable\ndef __default__():\n    self.purchase_contract.received()\n\n    '
    a0 = w3.eth.accounts[0]
    c = get_contract(contract_code, value=2)
    buyer_contract = get_contract(buyer_contract_code, *[c.address])
    buyer_contract_address = buyer_contract.address
    (init_bal_a0, init_bal_buyer_contract) = (w3.eth.get_balance(a0), w3.eth.get_balance(buyer_contract_address))
    buyer_contract.start_purchase(transact={'value': 4, 'from': w3.eth.accounts[1], 'gas': 100000})
    assert c.unlocked() is False
    assert c.buyer() == buyer_contract_address
    buyer_contract.start_received(transact={'from': w3.eth.accounts[1], 'gas': 100000})
    assert w3.eth.get_balance(a0), w3.eth.get_balance(buyer_contract_address) == (init_bal_a0 + 1, init_bal_buyer_contract - 1)