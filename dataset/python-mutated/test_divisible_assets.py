import pytest
import random
from bigchaindb.common.exceptions import DoubleSpend

def test_single_in_single_own_single_out_single_own_create(alice, user_pk, b):
    if False:
        for i in range(10):
            print('nop')
    from bigchaindb.models import Transaction
    tx = Transaction.create([alice.public_key], [([user_pk], 100)], asset={'name': random.random()})
    tx_signed = tx.sign([alice.private_key])
    assert tx_signed.validate(b) == tx_signed
    assert len(tx_signed.outputs) == 1
    assert tx_signed.outputs[0].amount == 100
    assert len(tx_signed.inputs) == 1

def test_single_in_single_own_multiple_out_single_own_create(alice, user_pk, b):
    if False:
        while True:
            i = 10
    from bigchaindb.models import Transaction
    tx = Transaction.create([alice.public_key], [([user_pk], 50), ([user_pk], 50)], asset={'name': random.random()})
    tx_signed = tx.sign([alice.private_key])
    assert tx_signed.validate(b) == tx_signed
    assert len(tx_signed.outputs) == 2
    assert tx_signed.outputs[0].amount == 50
    assert tx_signed.outputs[1].amount == 50
    assert len(tx_signed.inputs) == 1

def test_single_in_single_own_single_out_multiple_own_create(alice, user_pk, b):
    if False:
        while True:
            i = 10
    from bigchaindb.models import Transaction
    tx = Transaction.create([alice.public_key], [([user_pk, user_pk], 100)], asset={'name': random.random()})
    tx_signed = tx.sign([alice.private_key])
    assert tx_signed.validate(b) == tx_signed
    assert len(tx_signed.outputs) == 1
    assert tx_signed.outputs[0].amount == 100
    output = tx_signed.outputs[0].to_dict()
    assert 'subconditions' in output['condition']['details']
    assert len(output['condition']['details']['subconditions']) == 2
    assert len(tx_signed.inputs) == 1

def test_single_in_single_own_multiple_out_mix_own_create(alice, user_pk, b):
    if False:
        i = 10
        return i + 15
    from bigchaindb.models import Transaction
    tx = Transaction.create([alice.public_key], [([user_pk], 50), ([user_pk, user_pk], 50)], asset={'name': random.random()})
    tx_signed = tx.sign([alice.private_key])
    assert tx_signed.validate(b) == tx_signed
    assert len(tx_signed.outputs) == 2
    assert tx_signed.outputs[0].amount == 50
    assert tx_signed.outputs[1].amount == 50
    output_cid1 = tx_signed.outputs[1].to_dict()
    assert 'subconditions' in output_cid1['condition']['details']
    assert len(output_cid1['condition']['details']['subconditions']) == 2
    assert len(tx_signed.inputs) == 1

def test_single_in_multiple_own_single_out_single_own_create(alice, b, user_pk, user_sk):
    if False:
        while True:
            i = 10
    from bigchaindb.models import Transaction
    from bigchaindb.common.transaction import _fulfillment_to_details
    tx = Transaction.create([alice.public_key, user_pk], [([user_pk], 100)], asset={'name': random.random()})
    tx_signed = tx.sign([alice.private_key, user_sk])
    assert tx_signed.validate(b) == tx_signed
    assert len(tx_signed.outputs) == 1
    assert tx_signed.outputs[0].amount == 100
    assert len(tx_signed.inputs) == 1
    ffill = _fulfillment_to_details(tx_signed.inputs[0].fulfillment)
    assert 'subconditions' in ffill
    assert len(ffill['subconditions']) == 2

def test_single_in_single_own_single_out_single_own_transfer(alice, b, user_pk, user_sk):
    if False:
        for i in range(10):
            print('nop')
    from bigchaindb.models import Transaction
    tx_create = Transaction.create([alice.public_key], [([user_pk], 100)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 100)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([user_sk])
    b.store_bulk_transactions([tx_create_signed])
    assert tx_transfer_signed.validate(b)
    assert len(tx_transfer_signed.outputs) == 1
    assert tx_transfer_signed.outputs[0].amount == 100
    assert len(tx_transfer_signed.inputs) == 1

def test_single_in_single_own_multiple_out_single_own_transfer(alice, b, user_pk, user_sk):
    if False:
        i = 10
        return i + 15
    from bigchaindb.models import Transaction
    tx_create = Transaction.create([alice.public_key], [([user_pk], 100)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 50), ([alice.public_key], 50)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([user_sk])
    b.store_bulk_transactions([tx_create_signed])
    assert tx_transfer_signed.validate(b) == tx_transfer_signed
    assert len(tx_transfer_signed.outputs) == 2
    assert tx_transfer_signed.outputs[0].amount == 50
    assert tx_transfer_signed.outputs[1].amount == 50
    assert len(tx_transfer_signed.inputs) == 1

def test_single_in_single_own_single_out_multiple_own_transfer(alice, b, user_pk, user_sk):
    if False:
        i = 10
        return i + 15
    from bigchaindb.models import Transaction
    tx_create = Transaction.create([alice.public_key], [([user_pk], 100)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key, alice.public_key], 100)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([user_sk])
    b.store_bulk_transactions([tx_create_signed])
    assert tx_transfer_signed.validate(b) == tx_transfer_signed
    assert len(tx_transfer_signed.outputs) == 1
    assert tx_transfer_signed.outputs[0].amount == 100
    condition = tx_transfer_signed.outputs[0].to_dict()
    assert 'subconditions' in condition['condition']['details']
    assert len(condition['condition']['details']['subconditions']) == 2
    assert len(tx_transfer_signed.inputs) == 1
    b.store_bulk_transactions([tx_transfer_signed])
    with pytest.raises(DoubleSpend):
        tx_transfer_signed.validate(b)

def test_single_in_single_own_multiple_out_mix_own_transfer(alice, b, user_pk, user_sk):
    if False:
        for i in range(10):
            print('nop')
    from bigchaindb.models import Transaction
    tx_create = Transaction.create([alice.public_key], [([user_pk], 100)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 50), ([alice.public_key, alice.public_key], 50)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([user_sk])
    b.store_bulk_transactions([tx_create_signed])
    assert tx_transfer_signed.validate(b) == tx_transfer_signed
    assert len(tx_transfer_signed.outputs) == 2
    assert tx_transfer_signed.outputs[0].amount == 50
    assert tx_transfer_signed.outputs[1].amount == 50
    output_cid1 = tx_transfer_signed.outputs[1].to_dict()
    assert 'subconditions' in output_cid1['condition']['details']
    assert len(output_cid1['condition']['details']['subconditions']) == 2
    assert len(tx_transfer_signed.inputs) == 1
    b.store_bulk_transactions([tx_transfer_signed])
    with pytest.raises(DoubleSpend):
        tx_transfer_signed.validate(b)

def test_single_in_multiple_own_single_out_single_own_transfer(alice, b, user_pk, user_sk):
    if False:
        i = 10
        return i + 15
    from bigchaindb.models import Transaction
    from bigchaindb.common.transaction import _fulfillment_to_details
    tx_create = Transaction.create([alice.public_key], [([alice.public_key, user_pk], 100)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 100)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([alice.private_key, user_sk])
    b.store_bulk_transactions([tx_create_signed])
    assert tx_transfer_signed.validate(b) == tx_transfer_signed
    assert len(tx_transfer_signed.outputs) == 1
    assert tx_transfer_signed.outputs[0].amount == 100
    assert len(tx_transfer_signed.inputs) == 1
    ffill = _fulfillment_to_details(tx_transfer_signed.inputs[0].fulfillment)
    assert 'subconditions' in ffill
    assert len(ffill['subconditions']) == 2
    b.store_bulk_transactions([tx_transfer_signed])
    with pytest.raises(DoubleSpend):
        tx_transfer_signed.validate(b)

def test_multiple_in_single_own_single_out_single_own_transfer(alice, b, user_pk, user_sk):
    if False:
        i = 10
        return i + 15
    from bigchaindb.models import Transaction
    tx_create = Transaction.create([alice.public_key], [([user_pk], 50), ([user_pk], 50)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 100)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([user_sk])
    b.store_bulk_transactions([tx_create_signed])
    assert tx_transfer_signed.validate(b)
    assert len(tx_transfer_signed.outputs) == 1
    assert tx_transfer_signed.outputs[0].amount == 100
    assert len(tx_transfer_signed.inputs) == 2
    b.store_bulk_transactions([tx_transfer_signed])
    with pytest.raises(DoubleSpend):
        tx_transfer_signed.validate(b)

def test_multiple_in_multiple_own_single_out_single_own_transfer(alice, b, user_pk, user_sk):
    if False:
        i = 10
        return i + 15
    from bigchaindb.models import Transaction
    from bigchaindb.common.transaction import _fulfillment_to_details
    tx_create = Transaction.create([alice.public_key], [([user_pk, alice.public_key], 50), ([user_pk, alice.public_key], 50)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 100)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([alice.private_key, user_sk])
    b.store_bulk_transactions([tx_create_signed])
    assert tx_transfer_signed.validate(b) == tx_transfer_signed
    assert len(tx_transfer_signed.outputs) == 1
    assert tx_transfer_signed.outputs[0].amount == 100
    assert len(tx_transfer_signed.inputs) == 2
    ffill_fid0 = _fulfillment_to_details(tx_transfer_signed.inputs[0].fulfillment)
    ffill_fid1 = _fulfillment_to_details(tx_transfer_signed.inputs[1].fulfillment)
    assert 'subconditions' in ffill_fid0
    assert 'subconditions' in ffill_fid1
    assert len(ffill_fid0['subconditions']) == 2
    assert len(ffill_fid1['subconditions']) == 2
    b.store_bulk_transactions([tx_transfer_signed])
    with pytest.raises(DoubleSpend):
        tx_transfer_signed.validate(b)

def test_muiltiple_in_mix_own_multiple_out_single_own_transfer(alice, b, user_pk, user_sk):
    if False:
        i = 10
        return i + 15
    from bigchaindb.models import Transaction
    from bigchaindb.common.transaction import _fulfillment_to_details
    tx_create = Transaction.create([alice.public_key], [([user_pk], 50), ([user_pk, alice.public_key], 50)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 100)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([alice.private_key, user_sk])
    b.store_bulk_transactions([tx_create_signed])
    assert tx_transfer_signed.validate(b) == tx_transfer_signed
    assert len(tx_transfer_signed.outputs) == 1
    assert tx_transfer_signed.outputs[0].amount == 100
    assert len(tx_transfer_signed.inputs) == 2
    ffill_fid0 = _fulfillment_to_details(tx_transfer_signed.inputs[0].fulfillment)
    ffill_fid1 = _fulfillment_to_details(tx_transfer_signed.inputs[1].fulfillment)
    assert 'subconditions' not in ffill_fid0
    assert 'subconditions' in ffill_fid1
    assert len(ffill_fid1['subconditions']) == 2
    b.store_bulk_transactions([tx_transfer_signed])
    with pytest.raises(DoubleSpend):
        tx_transfer_signed.validate(b)

def test_muiltiple_in_mix_own_multiple_out_mix_own_transfer(alice, b, user_pk, user_sk):
    if False:
        print('Hello World!')
    from bigchaindb.models import Transaction
    from bigchaindb.common.transaction import _fulfillment_to_details
    tx_create = Transaction.create([alice.public_key], [([user_pk], 50), ([user_pk, alice.public_key], 50)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 50), ([alice.public_key, user_pk], 50)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([alice.private_key, user_sk])
    b.store_bulk_transactions([tx_create_signed])
    assert tx_transfer_signed.validate(b) == tx_transfer_signed
    assert len(tx_transfer_signed.outputs) == 2
    assert tx_transfer_signed.outputs[0].amount == 50
    assert tx_transfer_signed.outputs[1].amount == 50
    assert len(tx_transfer_signed.inputs) == 2
    cond_cid0 = tx_transfer_signed.outputs[0].to_dict()
    cond_cid1 = tx_transfer_signed.outputs[1].to_dict()
    assert 'subconditions' not in cond_cid0['condition']['details']
    assert 'subconditions' in cond_cid1['condition']['details']
    assert len(cond_cid1['condition']['details']['subconditions']) == 2
    ffill_fid0 = _fulfillment_to_details(tx_transfer_signed.inputs[0].fulfillment)
    ffill_fid1 = _fulfillment_to_details(tx_transfer_signed.inputs[1].fulfillment)
    assert 'subconditions' not in ffill_fid0
    assert 'subconditions' in ffill_fid1
    assert len(ffill_fid1['subconditions']) == 2
    b.store_bulk_transactions([tx_transfer_signed])
    with pytest.raises(DoubleSpend):
        tx_transfer_signed.validate(b)

def test_multiple_in_different_transactions(alice, b, user_pk, user_sk):
    if False:
        while True:
            i = 10
    from bigchaindb.models import Transaction
    tx_create = Transaction.create([alice.public_key], [([user_pk], 50), ([alice.public_key], 50)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer1 = Transaction.transfer(tx_create.to_inputs([1]), [([user_pk], 50)], asset_id=tx_create.id)
    tx_transfer1_signed = tx_transfer1.sign([alice.private_key])
    tx_transfer2 = Transaction.transfer(tx_create.to_inputs([0]) + tx_transfer1.to_inputs([0]), [([alice.private_key], 100)], asset_id=tx_create.id)
    tx_transfer2_signed = tx_transfer2.sign([user_sk])
    b.store_bulk_transactions([tx_create_signed, tx_transfer1_signed])
    assert tx_transfer2_signed.validate(b) == tx_transfer2_signed
    assert len(tx_transfer2_signed.outputs) == 1
    assert tx_transfer2_signed.outputs[0].amount == 100
    assert len(tx_transfer2_signed.inputs) == 2
    fid0_input = tx_transfer2_signed.inputs[0].fulfills.txid
    fid1_input = tx_transfer2_signed.inputs[1].fulfills.txid
    assert fid0_input == tx_create.id
    assert fid1_input == tx_transfer1.id

def test_amount_error_transfer(alice, b, user_pk, user_sk):
    if False:
        return 10
    from bigchaindb.models import Transaction
    from bigchaindb.common.exceptions import AmountError
    tx_create = Transaction.create([alice.public_key], [([user_pk], 100)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    b.store_bulk_transactions([tx_create_signed])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 50)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([user_sk])
    with pytest.raises(AmountError):
        tx_transfer_signed.validate(b)
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 101)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([user_sk])
    with pytest.raises(AmountError):
        tx_transfer_signed.validate(b)

def test_threshold_same_public_key(alice, b, user_pk, user_sk):
    if False:
        print('Hello World!')
    from bigchaindb.models import Transaction
    tx_create = Transaction.create([alice.public_key], [([user_pk, user_pk], 100)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 100)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([user_sk, user_sk])
    b.store_bulk_transactions([tx_create_signed])
    assert tx_transfer_signed.validate(b) == tx_transfer_signed
    b.store_bulk_transactions([tx_transfer_signed])
    with pytest.raises(DoubleSpend):
        tx_transfer_signed.validate(b)

def test_sum_amount(alice, b, user_pk, user_sk):
    if False:
        print('Hello World!')
    from bigchaindb.models import Transaction
    tx_create = Transaction.create([alice.public_key], [([user_pk], 1), ([user_pk], 1), ([user_pk], 1)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 3)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([user_sk])
    b.store_bulk_transactions([tx_create_signed])
    assert tx_transfer_signed.validate(b) == tx_transfer_signed
    assert len(tx_transfer_signed.outputs) == 1
    assert tx_transfer_signed.outputs[0].amount == 3
    b.store_bulk_transactions([tx_transfer_signed])
    with pytest.raises(DoubleSpend):
        tx_transfer_signed.validate(b)

def test_divide(alice, b, user_pk, user_sk):
    if False:
        i = 10
        return i + 15
    from bigchaindb.models import Transaction
    tx_create = Transaction.create([alice.public_key], [([user_pk], 3)], asset={'name': random.random()})
    tx_create_signed = tx_create.sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx_create.to_inputs(), [([alice.public_key], 1), ([alice.public_key], 1), ([alice.public_key], 1)], asset_id=tx_create.id)
    tx_transfer_signed = tx_transfer.sign([user_sk])
    b.store_bulk_transactions([tx_create_signed])
    assert tx_transfer_signed.validate(b) == tx_transfer_signed
    assert len(tx_transfer_signed.outputs) == 3
    for output in tx_transfer_signed.outputs:
        assert output.amount == 1
    b.store_bulk_transactions([tx_transfer_signed])
    with pytest.raises(DoubleSpend):
        tx_transfer_signed.validate(b)