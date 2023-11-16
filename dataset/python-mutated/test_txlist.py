"""Test getting a list of transactions from the backend.

This test module defines it's own fixture which is used by all the tests.
"""
import pytest

@pytest.fixture
def txlist(b, user_pk, user2_pk, user_sk, user2_sk):
    if False:
        print('Hello World!')
    from bigchaindb.models import Transaction
    create1 = Transaction.create([user_pk], [([user2_pk], 6)]).sign([user_sk])
    create2 = Transaction.create([user2_pk], [([user2_pk], 5), ([user_pk], 5)]).sign([user2_sk])
    transfer1 = Transaction.transfer(create1.to_inputs(), [([user_pk], 8)], create1.id).sign([user2_sk])
    b.store_bulk_transactions([create1, create2, transfer1])
    return type('', (), {'create1': create1, 'transfer1': transfer1})

@pytest.mark.bdb
def test_get_txlist_by_asset(b, txlist):
    if False:
        while True:
            i = 10
    res = b.get_transactions_filtered(txlist.create1.id)
    assert sorted(set((tx.id for tx in res))) == sorted(set([txlist.transfer1.id, txlist.create1.id]))

@pytest.mark.bdb
def test_get_txlist_by_operation(b, txlist):
    if False:
        print('Hello World!')
    res = b.get_transactions_filtered(txlist.create1.id, operation='CREATE')
    assert set((tx.id for tx in res)) == {txlist.create1.id}