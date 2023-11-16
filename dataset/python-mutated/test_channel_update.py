import json
from datetime import timedelta
from unittest.mock import call, patch
import graphene
import pytest
from django.utils.functional import SimpleLazyObject
from django.utils.text import slugify
from freezegun import freeze_time
from .....channel.error_codes import ChannelErrorCode
from .....core.utils.json_serializer import CustomJsonEncoder
from .....webhook.event_types import WebhookEventAsyncType
from .....webhook.payloads import generate_meta, generate_requestor
from ....tests.utils import assert_no_permission, get_graphql_content
from ...enums import AllocationStrategyEnum, MarkAsPaidStrategyEnum, TransactionFlowStrategyEnum
CHANNEL_UPDATE_MUTATION = '\n    mutation UpdateChannel($id: ID!,$input: ChannelUpdateInput!){\n        channelUpdate(id: $id, input: $input){\n            channel{\n                id\n                name\n                slug\n                currencyCode\n                defaultCountry {\n                    code\n                    country\n                }\n                warehouses {\n                    slug\n                }\n                stockSettings {\n                    allocationStrategy\n                }\n                orderSettings {\n                    automaticallyConfirmAllNewOrders\n                    automaticallyFulfillNonShippableGiftCard\n                    expireOrdersAfter\n                    markAsPaidStrategy\n                    deleteExpiredOrdersAfter\n                    allowUnpaidOrders\n                    includeDraftOrderInVoucherUsage\n                }\n            }\n            errors{\n                field\n                code\n                message\n                shippingZones\n                warehouses\n            }\n        }\n    }\n'

def test_channel_update_mutation_as_staff_user(permission_manage_channels, staff_api_client, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    default_country = 'FR'
    allocation_strategy = AllocationStrategyEnum.PRIORITIZE_SORTING_ORDER.name
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug, 'defaultCountry': default_country, 'stockSettings': {'allocationStrategy': allocation_strategy}, 'orderSettings': {'automaticallyConfirmAllNewOrders': False, 'automaticallyFulfillNonShippableGiftCard': False, 'expireOrdersAfter': 10, 'allowUnpaidOrders': True, 'includeDraftOrderInVoucherUsage': True}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    assert channel_data['name'] == channel_USD.name == name
    assert channel_data['slug'] == channel_USD.slug == slug
    assert channel_data['currencyCode'] == channel_USD.currency_code == 'USD'
    assert channel_data['defaultCountry']['code'] == channel_USD.default_country.code == default_country
    assert channel_data['stockSettings']['allocationStrategy'] == allocation_strategy
    assert channel_data['orderSettings']['automaticallyConfirmAllNewOrders'] is False
    assert channel_data['orderSettings']['automaticallyFulfillNonShippableGiftCard'] is False
    assert channel_data['orderSettings']['expireOrdersAfter'] == 10
    assert channel_data['orderSettings']['includeDraftOrderInVoucherUsage'] is True
    assert channel_data['orderSettings']['allowUnpaidOrders'] is True

def test_channel_update_mutation_as_app(permission_manage_channels, app_api_client, channel_USD):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug}}
    response = app_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    assert channel_data['name'] == channel_USD.name == name
    assert channel_data['slug'] == channel_USD.slug == slug
    assert channel_data['currencyCode'] == channel_USD.currency_code == 'USD'

def test_channel_update_mutation_as_customer(user_api_client, channel_USD):
    if False:
        i = 10
        return i + 15
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug}}
    response = user_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=())
    assert_no_permission(response)

def test_channel_update_mutation_as_anonymous(api_client, channel_USD):
    if False:
        i = 10
        return i + 15
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug}}
    response = api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=())
    assert_no_permission(response)

def test_channel_update_mutation_slugify_slug_field(permission_manage_channels, staff_api_client, channel_USD):
    if False:
        print('Hello World!')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'testName'
    slug = 'Invalid slug'
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    channel_data = content['data']['channelUpdate']['channel']
    assert channel_data['slug'] == slugify(slug)

def test_channel_update_mutation_with_duplicated_slug(permission_manage_channels, staff_api_client, channel_USD, channel_PLN):
    if False:
        i = 10
        return i + 15
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'New Channel'
    slug = channel_PLN.slug
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    error = content['data']['channelUpdate']['errors'][0]
    assert error['field'] == 'slug'
    assert error['code'] == ChannelErrorCode.UNIQUE.name

def test_channel_update_mutation_only_name(permission_manage_channels, staff_api_client, channel_USD):
    if False:
        return 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = channel_USD.slug
    variables = {'id': channel_id, 'input': {'name': name}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    assert channel_data['name'] == channel_USD.name == name
    assert channel_data['slug'] == channel_USD.slug == slug
    assert channel_data['currencyCode'] == channel_USD.currency_code == 'USD'

def test_channel_update_mutation_only_slug(permission_manage_channels, staff_api_client, channel_USD):
    if False:
        return 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = channel_USD.name
    slug = 'new_slug'
    variables = {'id': channel_id, 'input': {'slug': slug}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    assert channel_data['name'] == channel_USD.name == name
    assert channel_data['slug'] == channel_USD.slug == slug
    assert channel_data['currencyCode'] == channel_USD.currency_code == 'USD'

def test_channel_update_mutation_add_shipping_zone(permission_manage_channels, staff_api_client, channel_USD, shipping_zone):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug, 'addShippingZones': [shipping_zone_id]}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    shipping_zone.refresh_from_db()
    assert channel_data['name'] == channel_USD.name == name
    assert channel_data['slug'] == channel_USD.slug == slug
    assert channel_data['currencyCode'] == channel_USD.currency_code == 'USD'
    actual_shipping_zone = channel_USD.shipping_zones.first()
    assert actual_shipping_zone == shipping_zone

@patch('saleor.graphql.channel.mutations.channel_update.drop_invalid_shipping_methods_relations_for_given_channels.delay')
def test_channel_update_mutation_remove_shipping_zone(mocked_drop_invalid_shipping_methods_relations, permission_manage_channels, staff_api_client, channel_USD, shipping_zones, warehouses, channel_PLN):
    if False:
        return 10
    channel_USD.shipping_zones.add(*shipping_zones)
    channel_PLN.shipping_zones.add(*shipping_zones)
    for warehouse in warehouses:
        warehouse.shipping_zones.add(*shipping_zones)
    channel_PLN.warehouses.add(warehouses[0])
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    shipping_zone = shipping_zones[0]
    shipping_method_ids = shipping_zone.shipping_methods.values_list('id', flat=True)
    remove_shipping_zone = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug, 'removeShippingZones': [remove_shipping_zone]}}
    assert channel_USD.shipping_method_listings.filter(shipping_method__shipping_zone=shipping_zone)
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    assert channel_data['name'] == channel_USD.name == name
    assert channel_data['slug'] == channel_USD.slug == slug
    assert channel_data['currencyCode'] == channel_USD.currency_code == 'USD'
    assert not channel_USD.shipping_method_listings.filter(shipping_method__shipping_zone=shipping_zone)
    mocked_drop_invalid_shipping_methods_relations.assert_called_once_with(list(shipping_method_ids), [channel_USD.id])
    assert warehouses[0].id not in shipping_zones[0].warehouses.values('id')
    for zone in shipping_zones[1:]:
        assert zone.warehouses.count() == len(warehouses)

def test_channel_update_mutation_add_and_remove_shipping_zone(permission_manage_channels, staff_api_client, channel_USD, shipping_zones, shipping_zone):
    if False:
        while True:
            i = 10
    channel_USD.shipping_zones.add(*shipping_zones)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    remove_shipping_zone = graphene.Node.to_global_id('ShippingZone', shipping_zones[0].pk)
    add_shipping_zone = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug, 'addShippingZones': [add_shipping_zone], 'removeShippingZones': [remove_shipping_zone]}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    assert channel_data['name'] == channel_USD.name == name
    assert channel_data['slug'] == channel_USD.slug == slug
    assert channel_data['currencyCode'] == channel_USD.currency_code == 'USD'
    zones = channel_USD.shipping_zones.all()
    assert len(zones) == len(shipping_zones)

def test_channel_update_mutation_duplicated_shipping_zone(permission_manage_channels, staff_api_client, channel_USD, shipping_zones, shipping_zone):
    if False:
        for i in range(10):
            print('nop')
    channel_USD.shipping_zones.add(*shipping_zones)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    remove_shipping_zone = graphene.Node.to_global_id('ShippingZone', shipping_zones[0].pk)
    add_shipping_zone = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug, 'addShippingZones': [add_shipping_zone], 'removeShippingZones': [remove_shipping_zone, add_shipping_zone]}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['channel']
    errors = data['errors']
    assert len(errors) == 1
    assert errors[0]['field'] == 'shippingZones'
    assert errors[0]['code'] == ChannelErrorCode.DUPLICATED_INPUT_ITEM.name
    assert errors[0]['shippingZones'] == [add_shipping_zone]

@freeze_time('2022-05-12 12:00:00')
@patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_channel_update_mutation_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, permission_manage_channels, staff_api_client, channel_USD, settings):
    if False:
        print('Hello World!')
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    default_country = 'FR'
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug, 'defaultCountry': default_country, 'metadata': [{'key': 'key', 'value': 'value'}]}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    assert data['channel']
    update_webhook_args = [json.dumps({'id': channel_id, 'is_active': channel_USD.is_active, 'meta': generate_meta(requestor_data=generate_requestor(SimpleLazyObject(lambda : staff_api_client.user)))}, cls=CustomJsonEncoder), WebhookEventAsyncType.CHANNEL_UPDATED, [any_webhook], channel_USD, SimpleLazyObject(lambda : staff_api_client.user)]
    metadata_webhook_args = update_webhook_args.copy()
    metadata_webhook_args[1] = WebhookEventAsyncType.CHANNEL_METADATA_UPDATED
    mocked_webhook_trigger.assert_has_calls([call(*update_webhook_args), call(*metadata_webhook_args)])

def test_channel_update_mutation_add_warehouse(permission_manage_channels, staff_api_client, channel_USD, warehouse):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    warehouse_id = graphene.Node.to_global_id('Warehouse', warehouse.pk)
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug, 'addWarehouses': [warehouse_id]}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    warehouse.refresh_from_db()
    assert channel_data['name'] == channel_USD.name == name
    assert channel_data['slug'] == channel_USD.slug == slug
    assert channel_data['currencyCode'] == channel_USD.currency_code == 'USD'
    assert len(channel_data['warehouses']) == 1
    assert channel_data['warehouses'][0]['slug'] == warehouse.slug

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_channel_update_mutation_remove_warehouse(permission_manage_channels, staff_api_client, channel_USD, channel_PLN, channel_JPY, warehouses, warehouse_JPY, shipping_zones, count_queries):
    if False:
        i = 10
        return i + 15
    "Ensure that removing warehouses from channel works properly.\n\n    Also, ensure that when the warehouse is removed from the channel it's also removed\n    from shipping zones with which the warehouse do not have a common channel anymore.\n    "
    channel_USD.warehouses.add(*warehouses + [warehouse_JPY])
    channel_PLN.warehouses.add(*[warehouses[0], warehouse_JPY])
    channel_JPY.warehouses.add(warehouses[1])
    for shipping_zone in shipping_zones:
        shipping_zone.warehouses.add(*warehouses)
    shipping_zones[0].channels.add(channel_JPY)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    remove_warehouses = [graphene.Node.to_global_id('Warehouse', warehouse.pk) for warehouse in warehouses]
    warehouses_count = channel_USD.warehouses.count()
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug, 'removeWarehouses': remove_warehouses}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    assert channel_data['name'] == channel_USD.name == name
    assert channel_data['slug'] == channel_USD.slug == slug
    assert channel_data['currencyCode'] == channel_USD.currency_code == 'USD'
    assert len(channel_data['warehouses']) == warehouses_count - 2
    assert {warehouse_data['slug'] for warehouse_data in channel_data['warehouses']} == {warehouse_JPY.slug}
    for zone in shipping_zones:
        zone.refresh_from_db()
        assert warehouses[0] in zone.warehouses.all()
    assert warehouses[1] in shipping_zones[0].warehouses.all()
    assert warehouses[1] not in shipping_zones[1].warehouses.all()

def test_channel_update_mutation_add_and_remove_warehouse(permission_manage_channels, staff_api_client, channel_USD, warehouses, warehouse):
    if False:
        for i in range(10):
            print('nop')
    channel_USD.warehouses.add(*warehouses)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    remove_warehouse = graphene.Node.to_global_id('Warehouse', warehouses[0].pk)
    add_warehouse = graphene.Node.to_global_id('Warehouse', warehouse.pk)
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug, 'addWarehouses': [add_warehouse], 'removeWarehouses': [remove_warehouse]}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    assert channel_data['name'] == channel_USD.name == name
    assert channel_data['slug'] == channel_USD.slug == slug
    assert channel_data['currencyCode'] == channel_USD.currency_code == 'USD'
    assert {warehouse_data['slug'] for warehouse_data in channel_data['warehouses']} == {warehouse.slug for warehouse in warehouses[1:] + [warehouse]}

def test_channel_update_mutation_duplicated_warehouses(permission_manage_channels, staff_api_client, channel_USD, warehouses, warehouse):
    if False:
        i = 10
        return i + 15
    channel_USD.warehouses.add(*warehouses)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    name = 'newName'
    slug = 'new_slug'
    remove_warehouse = graphene.Node.to_global_id('Warehouse', warehouses[0].pk)
    add_warehouse = graphene.Node.to_global_id('Warehouse', warehouse.pk)
    variables = {'id': channel_id, 'input': {'name': name, 'slug': slug, 'addWarehouses': [add_warehouse], 'removeWarehouses': [remove_warehouse, add_warehouse]}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['channel']
    errors = data['errors']
    assert len(errors) == 1
    assert errors[0]['field'] == 'warehouses'
    assert errors[0]['code'] == ChannelErrorCode.DUPLICATED_INPUT_ITEM.name
    assert errors[0]['warehouses'] == [add_warehouse]

@pytest.mark.parametrize('expire_input', [0, None])
def test_channel_update_mutation_disable_expire_orders(expire_input, permission_manage_channels, app_api_client, channel_USD):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_USD.expire_orders_after = 10
    channel_USD.save()
    variables = {'id': channel_id, 'input': {'orderSettings': {'expireOrdersAfter': expire_input}}}
    response = app_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    assert data['channel']['orderSettings']['expireOrdersAfter'] is None
    channel_USD.refresh_from_db()
    assert channel_USD.expire_orders_after is None

def test_channel_update_mutation_negative_expire_orders(permission_manage_channels, app_api_client, channel_USD):
    if False:
        print('Hello World!')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'orderSettings': {'expireOrdersAfter': -1}}}
    response = app_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    error = content['data']['channelUpdate']['errors'][0]
    assert error['field'] == 'expireOrdersAfter'
    assert error['code'] == ChannelErrorCode.INVALID.name

def test_channel_update_order_settings_manage_orders(permission_manage_orders, staff_api_client, channel_USD):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_USD.allow_unpaid_orders = True
    channel_USD.save()
    variables = {'id': channel_id, 'input': {'orderSettings': {'automaticallyConfirmAllNewOrders': False, 'automaticallyFulfillNonShippableGiftCard': False, 'allowUnpaidOrders': False}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_orders,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    assert channel_data['orderSettings']['automaticallyConfirmAllNewOrders'] is False
    assert channel_data['orderSettings']['automaticallyFulfillNonShippableGiftCard'] is False
    assert channel_data['orderSettings']['allowUnpaidOrders'] is False

def test_channel_update_order_settings_empty_order_settings(permission_manage_orders, staff_api_client, channel_USD):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_USD.expire_orders_after = 10
    channel_USD.save()
    variables = {'id': channel_id, 'input': {'orderSettings': {}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_orders,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    assert channel_data['orderSettings']['automaticallyConfirmAllNewOrders'] is True
    assert channel_data['orderSettings']['automaticallyFulfillNonShippableGiftCard'] is True
    assert channel_data['orderSettings']['expireOrdersAfter'] == 10
    channel_USD.refresh_from_db()
    assert channel_USD.automatically_confirm_all_new_orders is True
    assert channel_USD.automatically_fulfill_non_shippable_gift_card is True
    assert channel_USD.expire_orders_after == 10

def test_channel_update_order_settings_manage_orders_as_app(permission_manage_orders, app_api_client, channel_USD):
    if False:
        print('Hello World!')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'orderSettings': {'automaticallyConfirmAllNewOrders': False, 'automaticallyFulfillNonShippableGiftCard': False}}}
    response = app_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_orders,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    assert channel_data['orderSettings']['automaticallyConfirmAllNewOrders'] is False
    assert channel_data['orderSettings']['automaticallyFulfillNonShippableGiftCard'] is False

def test_channel_update_order_settings_manage_orders_permission_denied(permission_manage_orders, staff_api_client, channel_USD):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'name': 'newNAme', 'orderSettings': {'automaticallyConfirmAllNewOrders': False, 'automaticallyFulfillNonShippableGiftCard': False}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_orders,))
    assert_no_permission(response)

def test_channel_update_order_settings_manage_orders_as_app_permission_denied(permission_manage_orders, app_api_client, channel_USD):
    if False:
        print('Hello World!')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'name': 'newNAme', 'orderSettings': {'automaticallyConfirmAllNewOrders': False, 'automaticallyFulfillNonShippableGiftCard': False}}}
    response = app_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_orders,))
    assert_no_permission(response)

def test_channel_update_order_mark_as_paid_strategy(permission_manage_orders, staff_api_client, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'orderSettings': {'automaticallyConfirmAllNewOrders': False, 'automaticallyFulfillNonShippableGiftCard': False, 'markAsPaidStrategy': MarkAsPaidStrategyEnum.TRANSACTION_FLOW.name}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_orders,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    assert channel_data['orderSettings']['markAsPaidStrategy'] == MarkAsPaidStrategyEnum.TRANSACTION_FLOW.name
    channel_USD.refresh_from_db()
    assert channel_USD.order_mark_as_paid_strategy == MarkAsPaidStrategyEnum.TRANSACTION_FLOW.value

def test_channel_update_delete_expired_orders_after(permission_manage_orders, staff_api_client, channel_USD):
    if False:
        return 10
    channel_USD.delete_expired_orders_after = timedelta(days=1)
    channel_USD.save()
    delete_expired_after = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'orderSettings': {'deleteExpiredOrdersAfter': delete_expired_after}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_orders,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    assert channel_data['orderSettings']['deleteExpiredOrdersAfter'] == delete_expired_after
    assert channel_USD.delete_expired_orders_after == timedelta(days=delete_expired_after)

@pytest.mark.parametrize('delete_expired_after', [-1, 0, 121, 300])
def test_channel_update_set_incorrect_delete_expired_orders_after(delete_expired_after, permission_manage_orders, staff_api_client, channel_USD):
    if False:
        i = 10
        return i + 15
    channel_USD.delete_expired_orders_after = timedelta(days=1)
    channel_USD.save()
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'orderSettings': {'deleteExpiredOrdersAfter': delete_expired_after}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_orders,))
    content = get_graphql_content(response)
    error = content['data']['channelUpdate']['errors'][0]
    assert error['field'] == 'deleteExpiredOrdersAfter'
    assert error['code'] == ChannelErrorCode.INVALID.name

@patch('saleor.discount.tasks.decrease_voucher_codes_usage_task.delay')
def test_channel_update_order_settings_voucher_usage_disable(decrease_voucher_codes_usage_task_mock, permission_manage_orders, staff_api_client, channel_USD, draft_order_list_with_multiple_use_voucher):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_USD.include_draft_order_in_voucher_usage = True
    channel_USD.save(update_fields=['include_draft_order_in_voucher_usage'])
    variables = {'id': channel_id, 'input': {'orderSettings': {'includeDraftOrderInVoucherUsage': False}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_orders,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    assert data['channel']['orderSettings']['includeDraftOrderInVoucherUsage'] is False
    decrease_voucher_codes_usage_task_mock.assert_called_once()

@patch('saleor.discount.tasks.disconnect_voucher_codes_from_draft_orders_task.delay')
def test_channel_update_order_settings_voucher_usage_enable(disconnect_voucher_codes_from_draft_orders_task_mock, permission_manage_orders, staff_api_client, channel_USD, draft_order_list_with_multiple_use_voucher):
    if False:
        for i in range(10):
            print('nop')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_USD.include_draft_order_in_voucher_usage = False
    channel_USD.save(update_fields=['include_draft_order_in_voucher_usage'])
    variables = {'id': channel_id, 'input': {'orderSettings': {'includeDraftOrderInVoucherUsage': True}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_orders,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    assert data['channel']['orderSettings']['includeDraftOrderInVoucherUsage'] is True
    disconnect_voucher_codes_from_draft_orders_task_mock.assert_called_once()
CHANNEL_UPDATE_MUTATION_WITH_CHECKOUT_SETTINGS = '\n    mutation UpdateChannel($id: ID!,$input: ChannelUpdateInput!){\n        channelUpdate(id: $id, input: $input){\n            channel{\n                id\n                name\n                slug\n                currencyCode\n                checkoutSettings {\n                    useLegacyErrorFlow\n                }\n            }\n            errors{\n                field\n                code\n                message\n                shippingZones\n                warehouses\n            }\n        }\n    }\n'

def test_channel_update_set_checkout_use_legacy_error_flow(permission_manage_channels, staff_api_client, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'checkoutSettings': {'useLegacyErrorFlow': False}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION_WITH_CHECKOUT_SETTINGS, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    assert channel_data['checkoutSettings']['useLegacyErrorFlow'] is False
    assert channel_USD.use_legacy_error_flow_for_checkout is False

def test_channel_update_set_checkout_use_legacy_error_flow_with_checkout_permission(permission_manage_checkouts, staff_api_client, channel_USD):
    if False:
        i = 10
        return i + 15
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'checkoutSettings': {'useLegacyErrorFlow': False}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION_WITH_CHECKOUT_SETTINGS, variables=variables, permissions=(permission_manage_checkouts,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    assert channel_data['checkoutSettings']['useLegacyErrorFlow'] is False
    assert channel_USD.use_legacy_error_flow_for_checkout is False

def test_channel_update_set_checkout_use_legacy_error_flow_without_permission(staff_api_client, channel_USD):
    if False:
        return 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'checkoutSettings': {'useLegacyErrorFlow': False}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION_WITH_CHECKOUT_SETTINGS, variables=variables)
    assert_no_permission(response)

def test_channel_update_checkout_and_order_settings_with_manage_orders(staff_api_client, channel_USD, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'checkoutSettings': {'useLegacyErrorFlow': False}, 'orderSettings': {'automaticallyConfirmAllNewOrders': False}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION_WITH_CHECKOUT_SETTINGS, variables=variables, permissions=[permission_manage_orders])
    assert_no_permission(response)

def test_channel_update_order_and_checkout_settings_with_manage_checkouts(staff_api_client, channel_USD, permission_manage_checkouts):
    if False:
        return 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'orderSettings': {'automaticallyConfirmAllNewOrders': False}, 'checkoutSettings': {'useLegacyErrorFlow': False}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION_WITH_CHECKOUT_SETTINGS, variables=variables, permissions=[permission_manage_checkouts])
    assert_no_permission(response)

def test_channel_update_with_order_and_checkout_settings(staff_api_client, channel_USD, permission_manage_checkouts, permission_manage_orders):
    if False:
        for i in range(10):
            print('nop')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'orderSettings': {'automaticallyConfirmAllNewOrders': False}, 'checkoutSettings': {'useLegacyErrorFlow': False}}}
    query = '\n    mutation UpdateChannel($id: ID!,$input: ChannelUpdateInput!){\n        channelUpdate(id: $id, input: $input){\n            channel{\n                id\n                name\n                slug\n                currencyCode\n                checkoutSettings {\n                    useLegacyErrorFlow\n                }\n                orderSettings {\n                    automaticallyConfirmAllNewOrders\n                }\n            }\n            errors{\n                field\n                code\n                message\n                shippingZones\n                warehouses\n            }\n        }\n    }\n    '
    response = staff_api_client.post_graphql(query, variables=variables, permissions=[permission_manage_checkouts, permission_manage_orders])
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    channel_USD.refresh_from_db()
    assert channel_data['checkoutSettings']['useLegacyErrorFlow'] is False
    assert channel_USD.use_legacy_error_flow_for_checkout is False
    assert channel_data['orderSettings']['automaticallyConfirmAllNewOrders'] is False
    assert channel_USD.automatically_confirm_all_new_orders is False
CHANNEL_UPDATE_MUTATION_WITH_PAYMENT_SETTINGS = '\n    mutation UpdateChannel($id: ID!,$input: ChannelUpdateInput!){\n        channelUpdate(id: $id, input: $input){\n            channel{\n                id\n                name\n                slug\n                currencyCode\n                paymentSettings {\n                    defaultTransactionFlowStrategy\n                }\n            }\n            errors{\n                field\n                code\n                message\n                shippingZones\n                warehouses\n            }\n        }\n    }\n'

def test_channel_update_default_transaction_flow_strategy(permission_manage_channels, staff_api_client, channel_USD):
    if False:
        return 10
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'paymentSettings': {'defaultTransactionFlowStrategy': TransactionFlowStrategyEnum.AUTHORIZATION.name}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION_WITH_PAYMENT_SETTINGS, variables=variables, permissions=(permission_manage_channels,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    assert channel_data['paymentSettings']['defaultTransactionFlowStrategy'] == TransactionFlowStrategyEnum.AUTHORIZATION.name
    channel_USD.refresh_from_db()
    assert channel_USD.default_transaction_flow_strategy == TransactionFlowStrategyEnum.AUTHORIZATION.value

def test_channel_update_default_transaction_flow_strategy_with_payment_permission(permission_manage_payments, staff_api_client, channel_USD):
    if False:
        i = 10
        return i + 15
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': channel_id, 'input': {'paymentSettings': {'defaultTransactionFlowStrategy': TransactionFlowStrategyEnum.AUTHORIZATION.name}}}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION_WITH_PAYMENT_SETTINGS, variables=variables, permissions=(permission_manage_payments,))
    content = get_graphql_content(response)
    data = content['data']['channelUpdate']
    assert not data['errors']
    channel_data = data['channel']
    assert channel_data['paymentSettings']['defaultTransactionFlowStrategy'] == TransactionFlowStrategyEnum.AUTHORIZATION.name
    channel_USD.refresh_from_db()
    assert channel_USD.default_transaction_flow_strategy == TransactionFlowStrategyEnum.AUTHORIZATION.value