import json
from decimal import Decimal
import graphene
from .....webhook.event_types import WebhookEventSyncType
from .....webhook.models import Webhook
from .....webhook.transport.asynchronous.transport import create_deliveries_for_subscriptions
PAYMENT_GATEWAY_INITIALIZE_SESSION = '\nsubscription {\n  event{\n    ...on PaymentGatewayInitializeSession{\n      data\n      sourceObject{\n        __typename\n        ... on Checkout{\n          id\n          totalPrice{\n            gross{\n              amount\n            }\n          }\n        }\n        ... on Order{\n          id\n        }\n      }\n    }\n  }\n}\n'

def test_payment_gateway_initialize_session_checkout_with_data(checkout, webhook_app, permission_manage_payments):
    if False:
        print('Hello World!')
    webhook_app.permissions.add(permission_manage_payments)
    webhook = Webhook.objects.create(name='Webhook', app=webhook_app, subscription_query=PAYMENT_GATEWAY_INITIALIZE_SESSION)
    event_type = WebhookEventSyncType.PAYMENT_GATEWAY_INITIALIZE_SESSION
    webhook.events.create(event_type=event_type)
    payload_data = {'some': 'json data'}
    amount = Decimal('10')
    delivery = create_deliveries_for_subscriptions(event_type, (checkout, payload_data, amount), [webhook])[0]
    checkout_id = graphene.Node.to_global_id('Checkout', checkout.pk)
    assert delivery.payload
    assert delivery.payload.payload
    assert json.loads(delivery.payload.payload) == {'data': payload_data, 'sourceObject': {'__typename': 'Checkout', 'id': checkout_id, 'totalPrice': {'gross': {'amount': 0.0}}}}

def test_payment_gateway_initialize_session_checkout_without_data(checkout, webhook_app, permission_manage_payments):
    if False:
        for i in range(10):
            print('nop')
    webhook_app.permissions.add(permission_manage_payments)
    webhook = Webhook.objects.create(name='Webhook', app=webhook_app, subscription_query=PAYMENT_GATEWAY_INITIALIZE_SESSION)
    event_type = WebhookEventSyncType.PAYMENT_GATEWAY_INITIALIZE_SESSION
    webhook.events.create(event_type=event_type)
    payload_data = None
    amount = Decimal('10')
    delivery = create_deliveries_for_subscriptions(event_type, (checkout, payload_data, amount), [webhook])[0]
    checkout_id = graphene.Node.to_global_id('Checkout', checkout.pk)
    assert delivery.payload
    assert delivery.payload.payload
    assert json.loads(delivery.payload.payload) == {'data': None, 'sourceObject': {'__typename': 'Checkout', 'id': checkout_id, 'totalPrice': {'gross': {'amount': 0.0}}}}

def test_payment_gateway_initialize_session_order_with_data(order, webhook_app, permission_manage_payments):
    if False:
        print('Hello World!')
    webhook_app.permissions.add(permission_manage_payments)
    webhook = Webhook.objects.create(name='Webhook', app=webhook_app, subscription_query=PAYMENT_GATEWAY_INITIALIZE_SESSION)
    event_type = WebhookEventSyncType.PAYMENT_GATEWAY_INITIALIZE_SESSION
    webhook.events.create(event_type=event_type)
    payload_data = {'some': 'json data'}
    amount = Decimal('10')
    delivery = create_deliveries_for_subscriptions(event_type, (order, payload_data, amount), [webhook])[0]
    order_id = graphene.Node.to_global_id('Order', order.pk)
    assert delivery.payload
    assert delivery.payload.payload
    assert json.loads(delivery.payload.payload) == {'data': payload_data, 'sourceObject': {'__typename': 'Order', 'id': order_id}}

def test_payment_gateway_initialize_session_order_without_data(order, webhook_app, permission_manage_payments):
    if False:
        print('Hello World!')
    webhook_app.permissions.add(permission_manage_payments)
    webhook = Webhook.objects.create(name='Webhook', app=webhook_app, subscription_query=PAYMENT_GATEWAY_INITIALIZE_SESSION)
    event_type = WebhookEventSyncType.PAYMENT_GATEWAY_INITIALIZE_SESSION
    webhook.events.create(event_type=event_type)
    payload_data = None
    amount = Decimal('10')
    delivery = create_deliveries_for_subscriptions(event_type, (order, payload_data, amount), [webhook])[0]
    order_id = graphene.Node.to_global_id('Order', order.pk)
    assert delivery.payload
    assert delivery.payload.payload
    assert json.loads(delivery.payload.payload) == {'data': None, 'sourceObject': {'__typename': 'Order', 'id': order_id}}