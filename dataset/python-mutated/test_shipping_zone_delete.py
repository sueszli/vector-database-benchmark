import json
from unittest import mock
import graphene
import pytest
from django.utils.functional import SimpleLazyObject
from freezegun import freeze_time
from .....core.utils.json_serializer import CustomJsonEncoder
from .....webhook.event_types import WebhookEventAsyncType
from .....webhook.payloads import generate_meta, generate_requestor
from ....tests.utils import get_graphql_content
DELETE_SHIPPING_ZONE_MUTATION = '\n    mutation deleteShippingZone($id: ID!) {\n        shippingZoneDelete(id: $id) {\n            shippingZone {\n                id\n                name\n            }\n            errors {\n                field\n                code\n                message\n            }\n        }\n    }\n'

def test_delete_shipping_zone(staff_api_client, permission_manage_shipping, shipping_zone):
    if False:
        return 10
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    variables = {'id': shipping_zone_id}
    response = staff_api_client.post_graphql(DELETE_SHIPPING_ZONE_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingZoneDelete']['shippingZone']
    assert data['name'] == shipping_zone.name
    with pytest.raises(shipping_zone._meta.model.DoesNotExist):
        shipping_zone.refresh_from_db()

@freeze_time('2022-05-12 12:00:00')
@mock.patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_delete_shipping_zone_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, staff_api_client, permission_manage_shipping, shipping_zone, settings):
    if False:
        i = 10
        return i + 15
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    shipping_zone_id = shipping_zone.id
    variables = {'id': graphene.Node.to_global_id('ShippingZone', shipping_zone_id)}
    response = staff_api_client.post_graphql(DELETE_SHIPPING_ZONE_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingZoneDelete']
    assert content['data']['shippingZoneDelete']['shippingZone']
    assert data['errors'] == []
    mocked_webhook_trigger.assert_called_once_with(json.dumps({'id': data['shippingZone']['id'], 'meta': generate_meta(requestor_data=generate_requestor(SimpleLazyObject(lambda : staff_api_client.user)))}, cls=CustomJsonEncoder), WebhookEventAsyncType.SHIPPING_ZONE_DELETED, [any_webhook], shipping_zone, SimpleLazyObject(lambda : staff_api_client.user))