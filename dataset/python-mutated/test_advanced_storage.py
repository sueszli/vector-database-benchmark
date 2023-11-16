import pytest
from web3.exceptions import ValidationError
INITIAL_VALUE = 4

@pytest.fixture
def adv_storage_contract(w3, get_contract):
    if False:
        while True:
            i = 10
    with open('examples/storage/advanced_storage.vy') as f:
        contract_code = f.read()
        contract = get_contract(contract_code, INITIAL_VALUE)
    return contract

def test_initial_state(adv_storage_contract):
    if False:
        while True:
            i = 10
    assert adv_storage_contract.storedData() == INITIAL_VALUE

def test_failed_transactions(w3, adv_storage_contract, assert_tx_failed):
    if False:
        return 10
    k1 = w3.eth.accounts[1]
    assert_tx_failed(lambda : adv_storage_contract.set(-10, transact={'from': k1}))
    adv_storage_contract.set(150, transact={'from': k1})
    assert_tx_failed(lambda : adv_storage_contract.set(10, transact={'from': k1}))
    adv_storage_contract.reset(transact={'from': k1})
    adv_storage_contract.set(10, transact={'from': k1})
    assert adv_storage_contract.storedData() == 10
    assert_tx_failed(lambda : adv_storage_contract.set('foo', transact={'from': k1}), ValidationError)
    assert_tx_failed(lambda : adv_storage_contract.set(1, 2, transact={'from': k1}), ValidationError, 'invocation failed due to improper number of arguments')

def test_events(w3, adv_storage_contract, get_logs):
    if False:
        for i in range(10):
            print('nop')
    (k1, k2) = w3.eth.accounts[:2]
    tx1 = adv_storage_contract.set(10, transact={'from': k1})
    tx2 = adv_storage_contract.set(20, transact={'from': k2})
    tx3 = adv_storage_contract.reset(transact={'from': k1})
    logs1 = get_logs(tx1, adv_storage_contract, 'DataChange')
    logs2 = get_logs(tx2, adv_storage_contract, 'DataChange')
    logs3 = get_logs(tx3, adv_storage_contract, 'DataChange')
    assert len(logs1) == 1
    assert logs1[0].args.value == 10
    assert len(logs2) == 1
    assert logs2[0].args.setter == k2
    assert not logs3