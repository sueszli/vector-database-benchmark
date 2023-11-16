import json
import graphene
from .....webhook.event_types import WebhookEventSyncType
from .....webhook.transport.asynchronous.transport import create_deliveries_for_subscriptions
from .payloads import generate_payment_payload

def test_payment_authorize(payment, subscription_payment_authorize_webhook):
    if False:
        return 10
    webhooks = [subscription_payment_authorize_webhook]
    event_type = WebhookEventSyncType.PAYMENT_AUTHORIZE
    deliveries = create_deliveries_for_subscriptions(event_type, payment, webhooks)
    expected_payload = generate_payment_payload(payment)
    assert json.loads(deliveries[0].payload.payload) == expected_payload
    assert len(deliveries) == len(webhooks)
    assert deliveries[0].webhook == webhooks[0]

def test_payment_capture(payment, subscription_payment_capture_webhook):
    if False:
        return 10
    webhooks = [subscription_payment_capture_webhook]
    event_type = WebhookEventSyncType.PAYMENT_CAPTURE
    deliveries = create_deliveries_for_subscriptions(event_type, payment, webhooks)
    expected_payload = generate_payment_payload(payment)
    assert json.loads(deliveries[0].payload.payload) == expected_payload
    assert len(deliveries) == len(webhooks)
    assert deliveries[0].webhook == webhooks[0]

def test_payment_refund(payment, subscription_payment_refund_webhook):
    if False:
        while True:
            i = 10
    webhooks = [subscription_payment_refund_webhook]
    event_type = WebhookEventSyncType.PAYMENT_REFUND
    deliveries = create_deliveries_for_subscriptions(event_type, payment, webhooks)
    expected_payload = generate_payment_payload(payment)
    assert json.loads(deliveries[0].payload.payload) == expected_payload
    assert len(deliveries) == len(webhooks)
    assert deliveries[0].webhook == webhooks[0]

def test_payment_void(payment, subscription_payment_void_webhook):
    if False:
        i = 10
        return i + 15
    webhooks = [subscription_payment_void_webhook]
    event_type = WebhookEventSyncType.PAYMENT_VOID
    deliveries = create_deliveries_for_subscriptions(event_type, payment, webhooks)
    expected_payload = generate_payment_payload(payment)
    assert json.loads(deliveries[0].payload.payload) == expected_payload
    assert len(deliveries) == len(webhooks)
    assert deliveries[0].webhook == webhooks[0]

def test_payment_confirm(payment, subscription_payment_confirm_webhook):
    if False:
        while True:
            i = 10
    webhooks = [subscription_payment_confirm_webhook]
    event_type = WebhookEventSyncType.PAYMENT_CONFIRM
    deliveries = create_deliveries_for_subscriptions(event_type, payment, webhooks)
    expected_payload = generate_payment_payload(payment)
    assert json.loads(deliveries[0].payload.payload) == expected_payload
    assert len(deliveries) == len(webhooks)
    assert deliveries[0].webhook == webhooks[0]

def test_payment_process(payment, subscription_payment_process_webhook):
    if False:
        while True:
            i = 10
    webhooks = [subscription_payment_process_webhook]
    event_type = WebhookEventSyncType.PAYMENT_PROCESS
    deliveries = create_deliveries_for_subscriptions(event_type, payment, webhooks)
    expected_payload = generate_payment_payload(payment)
    assert json.loads(deliveries[0].payload.payload) == expected_payload
    assert len(deliveries) == len(webhooks)
    assert deliveries[0].webhook == webhooks[0]

def test_payment_list_gateways(checkout, subscription_payment_list_gateways_webhook):
    if False:
        while True:
            i = 10
    webhooks = [subscription_payment_list_gateways_webhook]
    event_type = WebhookEventSyncType.PAYMENT_LIST_GATEWAYS
    checkout_id = graphene.Node.to_global_id('Checkout', checkout.pk)
    deliveries = create_deliveries_for_subscriptions(event_type, checkout, webhooks)
    expected_payload = {'checkout': {'id': checkout_id}}
    assert json.loads(deliveries[0].payload.payload) == expected_payload
    assert len(deliveries) == len(webhooks)
    assert deliveries[0].webhook == webhooks[0]