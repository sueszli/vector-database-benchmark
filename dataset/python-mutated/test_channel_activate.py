import json
from unittest import mock
import graphene
from django.utils.functional import SimpleLazyObject
from freezegun import freeze_time
from .....channel.error_codes import ChannelErrorCode
from .....core.utils.json_serializer import CustomJsonEncoder
from .....webhook.event_types import WebhookEventAsyncType
from .....webhook.payloads import generate_meta, generate_requestor
from ....tests.utils import get_graphql_content
CHANNEL_ACTIVATE_MUTATION = '\n    mutation ActivateChannel($id: ID!) {\n        channelActivate(id: $id){\n            channel {\n                id\n                name\n                isActive\n            }\n            errors{\n                field\n                code\n                message\n            }\n        }\n    }\n'

def test_channel_activate_mutation(permission_manage_channels, staff_api_client, channel_USD):
    if False:
        return 10
    channel_USD.is_active = False
    channel_USD.save()
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id}
    response = staff_api_client.post_graphql(CHANNEL_ACTIVATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelActivate']
    assert not data['errors']
    assert data['channel']['name'] == channel_USD.name
    assert data['channel']['isActive'] is True

@freeze_time('2022-05-12 12:00:00')
@mock.patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_channel_activate_mutation_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, permission_manage_channels, staff_api_client, channel_USD, settings):
    if False:
        i = 10
        return i + 15
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    channel_USD.is_active = False
    channel_USD.save()
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id}
    response = staff_api_client.post_graphql(CHANNEL_ACTIVATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    assert not content['data']['channelActivate']['errors']
    mocked_webhook_trigger.assert_called_once_with(json.dumps({'id': variables['id'], 'is_active': True, 'meta': generate_meta(requestor_data=generate_requestor(SimpleLazyObject(lambda : staff_api_client.user)))}, cls=CustomJsonEncoder), WebhookEventAsyncType.CHANNEL_STATUS_CHANGED, [any_webhook], channel_USD, SimpleLazyObject(lambda : staff_api_client.user))

def test_channel_activate_mutation_on_activated_channel(permission_manage_channels, staff_api_client, channel_USD):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id}
    response = staff_api_client.post_graphql(CHANNEL_ACTIVATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelActivate']
    assert data['errors'][0]['field'] == 'id'
    assert data['errors'][0]['code'] == ChannelErrorCode.INVALID.name