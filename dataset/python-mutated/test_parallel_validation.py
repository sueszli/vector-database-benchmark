import pytest
from bigchaindb.common.crypto import generate_key_pair
from bigchaindb.models import Transaction
pytestmark = pytest.mark.tendermint

def generate_create_and_transfer(keypair=None):
    if False:
        while True:
            i = 10
    if not keypair:
        keypair = generate_key_pair()
    (priv_key, pub_key) = keypair
    create_tx = Transaction.create([pub_key], [([pub_key], 10)]).sign([priv_key])
    transfer_tx = Transaction.transfer(create_tx.to_inputs(), [([pub_key], 10)], asset_id=create_tx.id).sign([priv_key])
    return (create_tx, transfer_tx)

def test_validation_worker_process_multiple_transactions(b):
    if False:
        i = 10
        return i + 15
    import multiprocessing as mp
    from bigchaindb.parallel_validation import ValidationWorker, RESET, EXIT
    keypair = generate_key_pair()
    (create_tx, transfer_tx) = generate_create_and_transfer(keypair)
    double_spend = Transaction.transfer(create_tx.to_inputs(), [([keypair.public_key], 10)], asset_id=create_tx.id).sign([keypair.private_key])
    (in_queue, results_queue) = (mp.Queue(), mp.Queue())
    vw = ValidationWorker(in_queue, results_queue)
    in_queue.put((0, create_tx.to_dict()))
    in_queue.put((10, transfer_tx.to_dict()))
    in_queue.put((20, double_spend.to_dict()))
    in_queue.put(RESET)
    in_queue.put((0, create_tx.to_dict()))
    in_queue.put((5, transfer_tx.to_dict()))
    in_queue.put(RESET)
    in_queue.put((20, create_tx.to_dict()))
    in_queue.put((25, double_spend.to_dict()))
    in_queue.put((30, transfer_tx.to_dict()))
    in_queue.put(EXIT)
    vw.run()
    assert results_queue.get() == (0, create_tx)
    assert results_queue.get() == (10, transfer_tx)
    assert results_queue.get() == (20, False)
    assert results_queue.get() == (0, create_tx)
    assert results_queue.get() == (5, transfer_tx)
    assert results_queue.get() == (20, create_tx)
    assert results_queue.get() == (25, double_spend)
    assert results_queue.get() == (30, False)

def test_parallel_validator_routes_transactions_correctly(b, monkeypatch):
    if False:
        i = 10
        return i + 15
    import os
    from collections import defaultdict
    import multiprocessing as mp
    from json import dumps
    from bigchaindb.parallel_validation import ParallelValidator
    validation_called_by = mp.Queue()

    def validate(self, dict_transaction):
        if False:
            while True:
                i = 10
        validation_called_by.put((os.getpid(), dict_transaction['id']))
        return dict_transaction
    monkeypatch.setattr('bigchaindb.parallel_validation.ValidationWorker.validate', validate)
    transactions = [{'id': '0'}, {'id': '1'}, {'id': '2'}, {'id': '3'}]
    pv = ParallelValidator(number_of_workers=2)
    pv.start()
    for _ in range(2):
        for transaction in transactions:
            pv.validate(dumps(transaction).encode('utf8'))
        assert pv.result(timeout=1) == transactions
        worker_to_transactions = defaultdict(list)
        for _ in transactions:
            (worker_pid, transaction_id) = validation_called_by.get()
            worker_to_transactions[worker_pid].append(transaction_id)
        for (_, transaction_ids) in worker_to_transactions.items():
            assert len(transaction_ids) == 2
            assert all(filter(lambda x: int(x) % 2 == 0, transaction_ids)) or all(filter(lambda x: int(x) % 2 == 1, transaction_ids))
    pv.stop()