import os
import pytest
from bigchaindb_driver.exceptions import BadRequest
from bigchaindb_driver import BigchainDB
from bigchaindb_driver.crypto import generate_keypair

def test_divisible_assets():
    if False:
        print('Hello World!')
    bdb = BigchainDB(os.environ.get('BIGCHAINDB_ENDPOINT'))
    (alice, bob) = (generate_keypair(), generate_keypair())
    bike_token = {'data': {'token_for': {'bike': {'serial_number': 420420}}, 'description': 'Time share token. Each token equals one hour of riding.'}}
    prepared_token_tx = bdb.transactions.prepare(operation='CREATE', signers=alice.public_key, recipients=[([bob.public_key], 10)], asset=bike_token)
    fulfilled_token_tx = bdb.transactions.fulfill(prepared_token_tx, private_keys=alice.private_key)
    bdb.transactions.send_commit(fulfilled_token_tx)
    bike_token_id = fulfilled_token_tx['id']
    assert bdb.transactions.retrieve(bike_token_id), 'Cannot find transaction {}'.format(bike_token_id)
    assert bdb.transactions.retrieve(bike_token_id)['outputs'][0]['amount'] == '10'
    transfer_asset = {'id': bike_token_id}
    output_index = 0
    output = fulfilled_token_tx['outputs'][output_index]
    transfer_input = {'fulfillment': output['condition']['details'], 'fulfills': {'output_index': output_index, 'transaction_id': fulfilled_token_tx['id']}, 'owners_before': output['public_keys']}
    prepared_transfer_tx = bdb.transactions.prepare(operation='TRANSFER', asset=transfer_asset, inputs=transfer_input, recipients=[([alice.public_key], 3), ([bob.public_key], 7)])
    fulfilled_transfer_tx = bdb.transactions.fulfill(prepared_transfer_tx, private_keys=bob.private_key)
    sent_transfer_tx = bdb.transactions.send_commit(fulfilled_transfer_tx)
    assert bdb.transactions.retrieve(fulfilled_transfer_tx['id']) == sent_transfer_tx
    assert bdb.transactions.retrieve(fulfilled_transfer_tx['id'])['outputs'][0]['amount'] == '3'
    assert bdb.transactions.retrieve(fulfilled_transfer_tx['id'])['outputs'][1]['amount'] == '7'
    transfer_asset = {'id': bike_token_id}
    output_index = 1
    output = fulfilled_transfer_tx['outputs'][output_index]
    transfer_input = {'fulfillment': output['condition']['details'], 'fulfills': {'output_index': output_index, 'transaction_id': fulfilled_transfer_tx['id']}, 'owners_before': output['public_keys']}
    prepared_transfer_tx = bdb.transactions.prepare(operation='TRANSFER', asset=transfer_asset, inputs=transfer_input, recipients=[([alice.public_key], 8)])
    fulfilled_transfer_tx = bdb.transactions.fulfill(prepared_transfer_tx, private_keys=bob.private_key)
    with pytest.raises(BadRequest) as error:
        bdb.transactions.send_commit(fulfilled_transfer_tx)
    assert error.value.args[0] == 400
    message = 'Invalid transaction (AmountError): The amount used in the inputs `7` needs to be same as the amount used in the outputs `8`'
    assert error.value.args[2]['message'] == message