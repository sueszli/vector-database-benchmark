"""All tests of transaction structure. The concern here is that transaction
structural / schematic issues are caught when reading a transaction
(ie going from dict -> transaction).
"""
import json
import pytest
try:
    import hashlib as sha3
except ImportError:
    import sha3
from unittest.mock import MagicMock
from bigchaindb.common.exceptions import AmountError, SchemaValidationError, ThresholdTooDeep
from bigchaindb.models import Transaction

def validate(tx):
    if False:
        print('Hello World!')
    if isinstance(tx, Transaction):
        tx = tx.to_dict()
    Transaction.from_dict(tx)

def validate_raises(tx, exc=SchemaValidationError):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(exc):
        validate(tx)

def test_validation_passes(signed_create_tx):
    if False:
        i = 10
        return i + 15
    Transaction.from_dict(signed_create_tx.to_dict())

def test_tx_serialization_hash_function(signed_create_tx):
    if False:
        print('Hello World!')
    tx = signed_create_tx.to_dict()
    tx['id'] = None
    payload = json.dumps(tx, skipkeys=False, sort_keys=True, separators=(',', ':'))
    assert sha3.sha3_256(payload.encode()).hexdigest() == signed_create_tx.id

def test_tx_serialization_with_incorrect_hash(signed_create_tx):
    if False:
        i = 10
        return i + 15
    from bigchaindb.common.transaction import Transaction
    from bigchaindb.common.exceptions import InvalidHash
    tx = signed_create_tx.to_dict()
    tx['id'] = 'a' * 64
    with pytest.raises(InvalidHash):
        Transaction.validate_id(tx)

def test_tx_serialization_with_no_hash(signed_create_tx):
    if False:
        i = 10
        return i + 15
    from bigchaindb.common.exceptions import InvalidHash
    tx = signed_create_tx.to_dict()
    del tx['id']
    with pytest.raises(InvalidHash):
        Transaction.from_dict(tx)

def test_validate_invalid_operation(b, create_tx, alice):
    if False:
        return 10
    create_tx.operation = 'something invalid'
    signed_tx = create_tx.sign([alice.private_key])
    validate_raises(signed_tx)

def test_validate_fails_metadata_empty_dict(b, create_tx, alice):
    if False:
        i = 10
        return i + 15
    create_tx.metadata = {'a': 1}
    signed_tx = create_tx.sign([alice.private_key])
    validate(signed_tx)
    create_tx._id = None
    create_tx.fulfillment = None
    create_tx.metadata = None
    signed_tx = create_tx.sign([alice.private_key])
    validate(signed_tx)
    create_tx._id = None
    create_tx.fulfillment = None
    create_tx.metadata = {}
    signed_tx = create_tx.sign([alice.private_key])
    validate_raises(signed_tx)

def test_transfer_asset_schema(user_sk, signed_transfer_tx):
    if False:
        return 10
    from bigchaindb.common.transaction import Transaction
    tx = signed_transfer_tx.to_dict()
    validate(tx)
    tx['id'] = None
    tx['asset']['data'] = {}
    tx = Transaction.from_dict(tx).sign([user_sk]).to_dict()
    validate_raises(tx)
    tx['id'] = None
    del tx['asset']['data']
    tx['asset']['id'] = 'b' * 63
    tx = Transaction.from_dict(tx).sign([user_sk]).to_dict()
    validate_raises(tx)

def test_create_tx_no_asset_id(b, create_tx, alice):
    if False:
        for i in range(10):
            print('nop')
    create_tx.asset['id'] = 'b' * 64
    signed_tx = create_tx.sign([alice.private_key])
    validate_raises(signed_tx)

def test_create_tx_asset_type(b, create_tx, alice):
    if False:
        print('Hello World!')
    create_tx.asset['data'] = 'a'
    signed_tx = create_tx.sign([alice.private_key])
    validate_raises(signed_tx)

def test_create_tx_no_asset_data(b, create_tx, alice):
    if False:
        print('Hello World!')
    tx_body = create_tx.to_dict()
    del tx_body['asset']['data']
    tx_serialized = json.dumps(tx_body, skipkeys=False, sort_keys=True, separators=(',', ':'))
    tx_body['id'] = sha3.sha3_256(tx_serialized.encode()).hexdigest()
    validate_raises(tx_body)

def test_no_inputs(b, create_tx, alice):
    if False:
        while True:
            i = 10
    create_tx.inputs = []
    signed_tx = create_tx.sign([alice.private_key])
    validate_raises(signed_tx)

def test_create_single_input(b, create_tx, alice):
    if False:
        for i in range(10):
            print('nop')
    from bigchaindb.common.transaction import Transaction
    tx = create_tx.to_dict()
    tx['inputs'] += tx['inputs']
    tx = Transaction.from_dict(tx).sign([alice.private_key]).to_dict()
    validate_raises(tx)
    tx['id'] = None
    tx['inputs'] = []
    tx = Transaction.from_dict(tx).sign([alice.private_key]).to_dict()
    validate_raises(tx)

def test_create_tx_no_fulfills(b, create_tx, alice):
    if False:
        return 10
    from bigchaindb.common.transaction import Transaction
    tx = create_tx.to_dict()
    tx['inputs'][0]['fulfills'] = {'transaction_id': 'a' * 64, 'output_index': 0}
    tx = Transaction.from_dict(tx).sign([alice.private_key]).to_dict()
    validate_raises(tx)

def test_transfer_has_inputs(user_sk, signed_transfer_tx, alice):
    if False:
        i = 10
        return i + 15
    signed_transfer_tx.inputs = []
    signed_transfer_tx._id = None
    signed_transfer_tx.sign([user_sk])
    validate_raises(signed_transfer_tx)

def test_low_amounts(b, user_sk, create_tx, signed_transfer_tx, alice):
    if False:
        while True:
            i = 10
    for (sk, tx) in [(alice.private_key, create_tx), (user_sk, signed_transfer_tx)]:
        tx.outputs[0].amount = 0
        tx._id = None
        tx.sign([sk])
        validate_raises(tx, AmountError)
        tx.outputs[0].amount = -1
        tx._id = None
        tx.sign([sk])
        validate_raises(tx)

def test_high_amounts(b, create_tx, alice):
    if False:
        while True:
            i = 10
    create_tx.outputs[0].amount = 10 ** 21
    create_tx.sign([alice.private_key])
    validate_raises(create_tx)
    create_tx.outputs[0].amount = 9 * 10 ** 18 + 1
    create_tx._id = None
    create_tx.sign([alice.private_key])
    validate_raises(create_tx, AmountError)
    create_tx.outputs[0].amount -= 1
    create_tx._id = None
    create_tx.sign([alice.private_key])
    validate(create_tx)

def test_handle_threshold_overflow():
    if False:
        while True:
            i = 10
    from bigchaindb.common import transaction
    cond = {'type': 'ed25519-sha-256', 'public_key': 'a' * 43}
    for i in range(1000):
        cond = {'type': 'threshold-sha-256', 'threshold': 1, 'subconditions': [cond]}
    with pytest.raises(ThresholdTooDeep):
        transaction._fulfillment_from_details(cond)

def test_unsupported_condition_type():
    if False:
        print('Hello World!')
    from bigchaindb.common import transaction
    from cryptoconditions.exceptions import UnsupportedTypeError
    with pytest.raises(UnsupportedTypeError):
        transaction._fulfillment_from_details({'type': 'a'})
    with pytest.raises(UnsupportedTypeError):
        transaction._fulfillment_to_details(MagicMock(type_name='a'))

def test_validate_version(b, create_tx, alice):
    if False:
        i = 10
        return i + 15
    create_tx.version = '2.0'
    create_tx.sign([alice.private_key])
    validate(create_tx)
    create_tx.version = '0.10'
    create_tx._id = None
    create_tx.sign([alice.private_key])
    validate_raises(create_tx)
    create_tx.version = '110'
    create_tx._id = None
    create_tx.sign([alice.private_key])
    validate_raises(create_tx)