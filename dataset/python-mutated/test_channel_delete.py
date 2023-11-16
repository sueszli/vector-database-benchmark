import json
from unittest import mock
import graphene
from django.utils.functional import SimpleLazyObject
from freezegun import freeze_time
from .....channel.error_codes import ChannelErrorCode
from .....channel.models import Channel
from .....checkout.models import Checkout
from .....core.utils.json_serializer import CustomJsonEncoder
from .....order.models import Order
from .....webhook.event_types import WebhookEventAsyncType
from .....webhook.payloads import generate_meta, generate_requestor
from ....tests.utils import assert_no_permission, get_graphql_content
CHANNEL_DELETE_MUTATION = '\n    mutation deleteChannel($id: ID!, $input: ChannelDeleteInput){\n        channelDelete(id: $id, input: $input){\n            channel{\n                id\n                name\n                slug\n                currencyCode\n            }\n            errors{\n                field\n                code\n                message\n            }\n        }\n    }\n'

def test_channel_delete_mutation_as_staff_user(order_list, checkout, permission_manage_channels, staff_api_client, channel_USD, other_channel_USD, product):
    if False:
        print('Hello World!')
    order = order_list[0]
    order.channel = channel_USD
    order.save()
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_target_id = graphene.Node.to_global_id('Channel', other_channel_USD.id)
    variables = {'id': channel_id, 'input': {'channelId': channel_target_id}}
    assert Checkout.objects.first() is not None
    response = staff_api_client.post_graphql(CHANNEL_DELETE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    get_graphql_content(response)
    order.refresh_from_db()
    assert order.channel == other_channel_USD
    assert Checkout.objects.first() is None
    assert not Channel.objects.filter(slug=channel_USD.slug).exists()

def test_channel_delete_mutation_with_the_same_channel_and_target_channel_id(permission_manage_channels, staff_api_client, channel_USD):
    if False:
        print('Hello World!')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'channelId': channel_id}}
    response = staff_api_client.post_graphql(CHANNEL_DELETE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    error = content['data']['channelDelete']['errors'][0]
    assert error['field'] == 'channelId'
    assert error['code'] == ChannelErrorCode.INVALID.name

def test_channel_delete_mutation_without_migration_channel_with_orders(permission_manage_channels, staff_api_client, channel_USD, checkout, order_list):
    if False:
        print('Hello World!')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id}
    checkout = Checkout.objects.first()
    assert checkout.channel == channel_USD
    assert Order.objects.filter(channel=channel_USD).exists()
    response = staff_api_client.post_graphql(CHANNEL_DELETE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    error = content['data']['channelDelete']['errors'][0]
    assert error['field'] == 'id'
    assert error['code'] == ChannelErrorCode.CHANNEL_WITH_ORDERS.name
    assert Channel.objects.filter(slug=channel_USD.slug).exists()

def test_channel_delete_mutation_without_orders_in_channel(permission_manage_channels, staff_api_client, channel_USD, checkout):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id}
    checkout = Checkout.objects.first()
    assert checkout.channel == channel_USD
    assert Order.objects.first() is None
    response = staff_api_client.post_graphql(CHANNEL_DELETE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    assert not content['data']['channelDelete']['errors']
    assert Checkout.objects.first() is None
    assert not Channel.objects.filter(slug=channel_USD.slug).exists()

def test_channel_delete_mutation_with_different_currency(permission_manage_channels, staff_api_client, channel_USD, channel_PLN):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    target_channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    variables = {'id': channel_id, 'input': {'channelId': target_channel_id}}
    response = staff_api_client.post_graphql(CHANNEL_DELETE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    error = content['data']['channelDelete']['errors'][0]
    assert error['field'] == 'channelId'
    assert error['code'] == ChannelErrorCode.CHANNELS_CURRENCY_MUST_BE_THE_SAME.name

def test_channel_delete_mutation_as_app(permission_manage_channels, app_api_client, order_list, channel_USD, other_channel_USD, checkout):
    if False:
        while True:
            i = 10
    order = order_list[0]
    order.channel = channel_USD
    order.save()
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_target_id = graphene.Node.to_global_id('Channel', other_channel_USD.id)
    variables = {'id': channel_id, 'input': {'channelId': channel_target_id}}
    assert Checkout.objects.first() is not None
    response = app_api_client.post_graphql(CHANNEL_DELETE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    get_graphql_content(response)
    order.refresh_from_db()
    assert order.channel == other_channel_USD
    assert Checkout.objects.first() is None
    assert not Channel.objects.filter(slug=channel_USD.slug).exists()

def test_channel_delete_mutation_as_customer(user_api_client, channel_USD, other_channel_USD):
    if False:
        i = 10
        return i + 15
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_target_id = graphene.Node.to_global_id('Channel', other_channel_USD.id)
    variables = {'id': channel_id, 'input': {'channelId': channel_target_id}}
    response = user_api_client.post_graphql(CHANNEL_DELETE_MUTATION, variables=variables, permissions=())
    assert_no_permission(response)
    assert Channel.objects.filter(slug=channel_USD.slug).exists()

def test_channel_delete_mutation_as_anonymous(api_client, channel_USD, other_channel_USD):
    if False:
        for i in range(10):
            print('nop')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_target_id = graphene.Node.to_global_id('Channel', other_channel_USD.id)
    variables = {'id': channel_id, 'input': {'channelId': channel_target_id}}
    response = api_client.post_graphql(CHANNEL_DELETE_MUTATION, variables=variables, permissions=())
    assert_no_permission(response)
    assert Channel.objects.filter(slug=channel_USD.slug).exists()

@freeze_time('2022-05-12 12:00:00')
@mock.patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_channel_delete_mutation_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, checkout, permission_manage_channels, staff_api_client, channel_USD, other_channel_USD, settings):
    if False:
        return 10
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_target_id = graphene.Node.to_global_id('Channel', other_channel_USD.id)
    variables = {'id': channel_id, 'input': {'channelId': channel_target_id}}
    assert Checkout.objects.first() is not None
    response = staff_api_client.post_graphql(CHANNEL_DELETE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    get_graphql_content(response)
    assert not Channel.objects.filter(slug=channel_USD.slug).exists()
    mocked_webhook_trigger.assert_called_once_with(json.dumps({'id': graphene.Node.to_global_id('Channel', channel_USD.id), 'is_active': channel_USD.is_active, 'meta': generate_meta(requestor_data=generate_requestor(SimpleLazyObject(lambda : staff_api_client.user)))}, cls=CustomJsonEncoder), WebhookEventAsyncType.CHANNEL_DELETED, [any_webhook], channel_USD, SimpleLazyObject(lambda : staff_api_client.user))

def test_channel_delete_mutation_deletes_invalid_warehouse_to_zone_relations(order_list, checkout, permission_manage_channels, staff_api_client, channel_USD, channel_PLN, channel_JPY, warehouses, warehouse_JPY, shipping_zones, other_channel_USD):
    if False:
        i = 10
        return i + 15
    'Ensure deleting channel deletes no longer valid warehouse to zone relations.'
    order = order_list[0]
    order.channel = channel_USD
    order.save()
    channel_USD.warehouses.add(*warehouses)
    channel_PLN.warehouses.add(warehouses[0])
    channel_JPY.warehouses.add(warehouses[1])
    for shipping_zone in shipping_zones:
        shipping_zone.warehouses.add(*warehouses)
    shipping_zones[0].channels.add(channel_JPY)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_target_id = graphene.Node.to_global_id('Channel', other_channel_USD.id)
    variables = {'id': channel_id, 'input': {'channelId': channel_target_id}}
    assert Checkout.objects.first() is not None
    response = staff_api_client.post_graphql(CHANNEL_DELETE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    get_graphql_content(response)
    order.refresh_from_db()
    assert order.channel == other_channel_USD
    assert Checkout.objects.first() is None
    assert not Channel.objects.filter(slug=channel_USD.slug).exists()
    for zone in shipping_zones:
        zone.refresh_from_db()
        assert warehouses[0] in zone.warehouses.all()
    assert warehouses[1] in shipping_zones[0].warehouses.all()
    assert warehouses[1] not in shipping_zones[1].warehouses.all()