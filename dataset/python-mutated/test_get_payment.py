import logging
import graphene
import pytest
from ...webhooks import get_payment
logger = logging.getLogger(__name__)

@pytest.mark.parametrize('payment_id', ['123', 'Test payment ID', 'ó À È Ì Ò Ù Ỳ'])
def test_get_payment_invalid_payment_id(payment_id, caplog):
    if False:
        print('Hello World!')
    caplog.set_level(logging.WARNING)
    result = get_payment(payment_id)
    assert result is None
    assert f'Unable to decode the payment ID {payment_id}.' in caplog.text

@pytest.mark.parametrize('payment_id', ['  ', None])
def test_get_payment_missing_payment_id(payment_id, caplog):
    if False:
        return 10
    caplog.set_level(logging.WARNING)
    result = get_payment(payment_id)
    assert result is None
    assert 'Missing payment ID.' in caplog.text

def test_get_payment_not_active_payment(payment_dummy, caplog):
    if False:
        for i in range(10):
            print('nop')
    caplog.set_level(logging.WARNING)
    payment_dummy.is_active = False
    payment_dummy.save(update_fields=['is_active'])
    payment_id = graphene.Node.to_global_id('Payment', payment_dummy.pk)
    transaction_id = 'psp reference'
    result = get_payment(payment_id, transaction_id)
    expected_msg = f'Payment for {payment_id} ({payment_dummy.pk}) was not found. Reference {transaction_id}'
    assert not result
    assert expected_msg in caplog.text

def test_get_payment_for_notification_related_to_transaction_item(payment_dummy):
    if False:
        for i in range(10):
            print('nop')
    incorrect_payment_id = graphene.Node.to_global_id('TransactionItem', payment_dummy.pk)
    transaction_id = 'psp reference'
    result = get_payment(incorrect_payment_id, transaction_id)
    assert not result