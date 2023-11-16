import os
from bigchaindb_driver import BigchainDB
from bigchaindb_driver.crypto import generate_keypair

def test_multiple_owners():
    if False:
        i = 10
        return i + 15
    bdb = BigchainDB(os.environ.get('BIGCHAINDB_ENDPOINT'))
    (alice, bob) = (generate_keypair(), generate_keypair())
    dw_asset = {'data': {'dish washer': {'serial_number': 1337}}}
    prepared_dw_tx = bdb.transactions.prepare(operation='CREATE', signers=alice.public_key, recipients=(alice.public_key, bob.public_key), asset=dw_asset)
    fulfilled_dw_tx = bdb.transactions.fulfill(prepared_dw_tx, private_keys=[alice.private_key, bob.private_key])
    bdb.transactions.send_commit(fulfilled_dw_tx)
    dw_id = fulfilled_dw_tx['id']
    assert bdb.transactions.retrieve(dw_id), 'Cannot find transaction {}'.format(dw_id)
    assert len(bdb.transactions.retrieve(dw_id)['outputs'][0]['public_keys']) == 2
    carol = generate_keypair()
    transfer_asset = {'id': dw_id}
    output_index = 0
    output = fulfilled_dw_tx['outputs'][output_index]
    transfer_input = {'fulfillment': output['condition']['details'], 'fulfills': {'output_index': output_index, 'transaction_id': fulfilled_dw_tx['id']}, 'owners_before': output['public_keys']}
    prepared_transfer_tx = bdb.transactions.prepare(operation='TRANSFER', asset=transfer_asset, inputs=transfer_input, recipients=carol.public_key)
    fulfilled_transfer_tx = bdb.transactions.fulfill(prepared_transfer_tx, private_keys=[alice.private_key, bob.private_key])
    sent_transfer_tx = bdb.transactions.send_commit(fulfilled_transfer_tx)
    assert bdb.transactions.retrieve(fulfilled_transfer_tx['id']) == sent_transfer_tx
    assert len(bdb.transactions.retrieve(fulfilled_transfer_tx['id'])['inputs'][0]['owners_before']) == 2
    assert bdb.transactions.retrieve(fulfilled_transfer_tx['id'])['outputs'][0]['public_keys'][0] == carol.public_key