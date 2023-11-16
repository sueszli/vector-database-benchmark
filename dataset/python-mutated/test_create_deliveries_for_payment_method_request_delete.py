import json
import graphene
from .....payment.interface import StoredPaymentMethodRequestDeleteData
from .....webhook.event_types import WebhookEventSyncType
from .....webhook.transport.asynchronous.transport import create_deliveries_for_subscriptions
STORED_PAYMENT_METHOD_DELETE_REQUESTED = '\nsubscription {\n  event {\n    ... on StoredPaymentMethodDeleteRequested{\n      user{\n        id\n      }\n      paymentMethodId\n      channel{\n        id\n      }\n    }\n  }\n}\n'

def test_stored_payment_method_request_delete(stored_payment_method_request_delete_app, customer_user, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    webhook = stored_payment_method_request_delete_app.webhooks.first()
    webhook.subscription_query = STORED_PAYMENT_METHOD_DELETE_REQUESTED
    webhook.save()
    payment_method_id = '123'
    request_delete_data = StoredPaymentMethodRequestDeleteData(user=customer_user, payment_method_id=payment_method_id, channel=channel_USD)
    event_type = WebhookEventSyncType.STORED_PAYMENT_METHOD_DELETE_REQUESTED
    delivery = create_deliveries_for_subscriptions(event_type, request_delete_data, [webhook])[0]
    assert delivery.payload
    assert delivery.payload.payload
    assert json.loads(delivery.payload.payload) == {'paymentMethodId': payment_method_id, 'user': {'id': graphene.Node.to_global_id('User', customer_user.pk)}, 'channel': {'id': graphene.Node.to_global_id('Channel', channel_USD.pk)}}