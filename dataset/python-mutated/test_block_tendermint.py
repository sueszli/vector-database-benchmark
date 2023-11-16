import pytest
from bigchaindb.models import Transaction
from bigchaindb.lib import Block
BLOCKS_ENDPOINT = '/api/v1/blocks/'

@pytest.mark.bdb
@pytest.mark.usefixtures('inputs')
def test_get_block_endpoint(b, client, alice):
    if False:
        for i in range(10):
            print('nop')
    import copy
    tx = Transaction.create([alice.public_key], [([alice.public_key], 1)], asset={'cycle': 'hero'})
    tx = tx.sign([alice.private_key])
    tx_dict = copy.deepcopy(tx.to_dict())
    b.store_bulk_transactions([tx])
    block = Block(app_hash='random_utxo', height=31, transactions=[tx.id])
    b.store_block(block._asdict())
    res = client.get(BLOCKS_ENDPOINT + str(block.height))
    expected_response = {'height': block.height, 'transactions': [tx_dict]}
    assert res.json == expected_response
    assert res.status_code == 200

@pytest.mark.bdb
@pytest.mark.usefixtures('inputs')
def test_get_block_returns_404_if_not_found(client):
    if False:
        print('Hello World!')
    res = client.get(BLOCKS_ENDPOINT + '123')
    assert res.status_code == 404
    res = client.get(BLOCKS_ENDPOINT + '123/')
    assert res.status_code == 404

@pytest.mark.bdb
def test_get_block_containing_transaction(b, client, alice):
    if False:
        i = 10
        return i + 15
    tx = Transaction.create([alice.public_key], [([alice.public_key], 1)], asset={'cycle': 'hero'})
    tx = tx.sign([alice.private_key])
    b.store_bulk_transactions([tx])
    block = Block(app_hash='random_utxo', height=13, transactions=[tx.id])
    b.store_block(block._asdict())
    res = client.get('{}?transaction_id={}'.format(BLOCKS_ENDPOINT, tx.id))
    expected_response = [block.height]
    assert res.json == expected_response
    assert res.status_code == 200

@pytest.mark.bdb
def test_get_blocks_by_txid_endpoint_returns_empty_list_not_found(client):
    if False:
        for i in range(10):
            print('nop')
    res = client.get(BLOCKS_ENDPOINT + '?transaction_id=')
    assert res.status_code == 200
    assert len(res.json) == 0
    res = client.get(BLOCKS_ENDPOINT + '?transaction_id=123')
    assert res.status_code == 200
    assert len(res.json) == 0