from datetime import timedelta
from unittest import mock
from unittest.mock import patch
import graphene
from django.utils import timezone
from freezegun import freeze_time
from ....checkout.fetch import fetch_checkout_info, fetch_checkout_lines
from ....checkout.utils import invalidate_checkout_prices
from ....plugins.manager import get_plugins_manager
from ...tests.utils import get_graphql_content
ADD_CHECKOUT_LINES = '\nmutation addCheckoutLines($checkoutId: ID!, $line: CheckoutLineInput!) {\n  checkoutLinesAdd(checkoutId: $checkoutId, lines: [$line]) {\n    errors {\n      field\n      message\n    }\n  }\n}\n'

@patch('saleor.graphql.checkout.mutations.checkout_lines_add.invalidate_checkout_prices')
def test_checkout_lines_add_invalidate_prices(mocked_function, api_client, checkout_with_items, stock):
    if False:
        return 10
    manager = get_plugins_manager()
    query = ADD_CHECKOUT_LINES
    variables = {'checkoutId': graphene.Node.to_global_id('Checkout', checkout_with_items.pk), 'line': {'quantity': 1, 'variantId': graphene.Node.to_global_id('ProductVariant', stock.product_variant.pk)}}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutLinesAdd']['errors']
    checkout_with_items.refresh_from_db()
    (lines, _) = fetch_checkout_lines(checkout_with_items)
    checkout_info = fetch_checkout_info(checkout_with_items, lines, manager)
    mocked_function.assert_called_once_with(checkout_info, lines, mock.ANY, save=True)
UPDATE_CHECKOUT_LINES = '\nmutation updateCheckoutLines($token: UUID!, $line: CheckoutLineUpdateInput!) {\n  checkoutLinesUpdate(token: $token, lines: [$line]) {\n    errors {\n      field\n      message\n    }\n  }\n}\n'

@patch('saleor.graphql.checkout.mutations.checkout_lines_add.invalidate_checkout_prices')
def test_checkout_lines_update_invalidate_prices(mocked_function, api_client, checkout_with_items, stock):
    if False:
        for i in range(10):
            print('nop')
    manager = get_plugins_manager()
    query = UPDATE_CHECKOUT_LINES
    variables = {'token': checkout_with_items.token, 'line': {'quantity': 1, 'variantId': graphene.Node.to_global_id('ProductVariant', stock.product_variant.pk)}}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutLinesUpdate']['errors']
    checkout_with_items.refresh_from_db()
    (lines, _) = fetch_checkout_lines(checkout_with_items)
    checkout_info = fetch_checkout_info(checkout_with_items, lines, manager)
    mocked_function.assert_called_once_with(checkout_info, lines, mock.ANY, save=True)
DELETE_CHECKOUT_LINES = '\nmutation deleteCheckoutLines($token: UUID!, $lineId: ID!){\n  checkoutLinesDelete(token: $token, linesIds: [$lineId]) {\n    errors {\n      field\n      message\n    }\n  }\n}\n'

@patch('saleor.graphql.checkout.mutations.checkout_lines_delete.invalidate_checkout_prices')
def test_checkout_lines_delete_invalidate_prices(mocked_function, api_client, checkout_with_items):
    if False:
        while True:
            i = 10
    manager = get_plugins_manager()
    query = DELETE_CHECKOUT_LINES
    variables = {'token': checkout_with_items.token, 'lineId': graphene.Node.to_global_id('CheckoutLine', checkout_with_items.lines.first().pk)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutLinesDelete']['errors']
    checkout_with_items.refresh_from_db()
    (lines, _) = fetch_checkout_lines(checkout_with_items)
    checkout_info = fetch_checkout_info(checkout_with_items, lines, manager)
    mocked_function.assert_called_once_with(checkout_info, lines, mock.ANY, save=True)
DELETE_CHECKOUT_LINE = '\nmutation deleteCheckoutLine($token: UUID!, $lineId: ID!){\n  checkoutLineDelete(token: $token, lineId: $lineId) {\n    errors {\n      field\n      message\n    }\n  }\n}\n'

@patch('saleor.graphql.checkout.mutations.checkout_line_delete.invalidate_checkout_prices')
def test_checkout_line_delete_invalidate_prices(mocked_function, api_client, checkout_with_items):
    if False:
        for i in range(10):
            print('nop')
    manager = get_plugins_manager()
    query = DELETE_CHECKOUT_LINE
    variables = {'token': checkout_with_items.token, 'lineId': graphene.Node.to_global_id('CheckoutLine', checkout_with_items.lines.first().pk)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutLineDelete']['errors']
    checkout_with_items.refresh_from_db()
    (lines, _) = fetch_checkout_lines(checkout_with_items)
    checkout_info = fetch_checkout_info(checkout_with_items, lines, manager)
    mocked_function.assert_called_once_with(checkout_info, lines, mock.ANY, save=True)
UPDATE_CHECKOUT_SHIPPING_ADDRESS = '\nmutation UpdateCheckoutShippingAddress($token: UUID!, $address: AddressInput!) {\n  checkoutShippingAddressUpdate(token: $token, shippingAddress: $address) {\n    errors {\n      field\n      message\n    }\n  }\n}\n'

@patch('saleor.graphql.checkout.mutations.checkout_shipping_address_update.invalidate_checkout_prices')
def test_checkout_shipping_address_update_invalidate_prices(mocked_function, api_client, checkout_with_items, graphql_address_data, plugins_manager):
    if False:
        return 10
    manager = get_plugins_manager()
    query = UPDATE_CHECKOUT_SHIPPING_ADDRESS
    variables = {'token': checkout_with_items.token, 'address': graphql_address_data}
    mocked_function.return_value = []
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutShippingAddressUpdate']['errors']
    checkout_with_items.refresh_from_db()
    (lines, _) = fetch_checkout_lines(checkout_with_items)
    checkout_info = fetch_checkout_info(checkout_with_items, lines, manager)
    mocked_function.assert_called_once_with(checkout_info, lines, mock.ANY, save=False)
UPDATE_CHECKOUT_BILLING_ADDRESS = '\nmutation UpdateCheckoutBillingAddress($token: UUID!, $address: AddressInput!) {\n  checkoutBillingAddressUpdate(token: $token, billingAddress: $address) {\n    errors {\n      field\n      message\n    }\n  }\n}\n'

@patch('saleor.graphql.checkout.mutations.checkout_billing_address_update.invalidate_checkout_prices')
def test_checkout_billing_address_update_invalidate_prices(mocked_function, api_client, checkout_with_items, graphql_address_data):
    if False:
        print('Hello World!')
    manager = get_plugins_manager()
    query = UPDATE_CHECKOUT_BILLING_ADDRESS
    variables = {'token': checkout_with_items.token, 'address': graphql_address_data}
    mocked_function.return_value = []
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutBillingAddressUpdate']['errors']
    checkout_with_items.refresh_from_db()
    (lines, _) = fetch_checkout_lines(checkout_with_items)
    checkout_info = fetch_checkout_info(checkout_with_items, lines, manager)
    mocked_function.assert_called_once_with(checkout_info, lines, mock.ANY, recalculate_discount=False, save=False)
UPDATE_CHECKOUT_SHIPPING_METHOD = '\nmutation updateCheckoutShippingOptions($token: UUID!, $shippingMethodId: ID) {\n  checkoutShippingMethodUpdate(token: $token, shippingMethodId: $shippingMethodId) {\n    errors {\n      field\n      message\n    }\n  }\n}\n'

@mock.patch('saleor.graphql.checkout.mutations.checkout_shipping_method_update.invalidate_checkout_prices', wraps=invalidate_checkout_prices)
def test_checkout_shipping_method_update_invalidate_prices(mocked_invalidate_checkout_prices, api_client, checkout_with_shipping_address, shipping_method):
    if False:
        return 10
    checkout = checkout_with_shipping_address
    checkout.price_expiration = timezone.now()
    checkout.save(update_fields=['price_expiration'])
    query = UPDATE_CHECKOUT_SHIPPING_METHOD
    variables = {'token': checkout.token, 'shippingMethodId': graphene.Node.to_global_id('ShippingMethod', shipping_method.pk)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    checkout.refresh_from_db()
    assert not response['data']['checkoutShippingMethodUpdate']['errors']
    assert mocked_invalidate_checkout_prices.call_count == 1
UPDATE_CHECKOUT_DELIVERY_METHOD = '\nmutation updateCheckoutDeliveryOptions($token: UUID!, $deliveryMethodId: ID!) {\n  checkoutDeliveryMethodUpdate(token: $token, deliveryMethodId: $deliveryMethodId) {\n    errors {\n      field\n      message\n    }\n  }\n}\n'

@mock.patch('saleor.graphql.checkout.mutations.checkout_delivery_method_update.invalidate_checkout_prices', wraps=invalidate_checkout_prices)
def test_checkout_delivery_method_update_invalidate_prices(mocked_invalidate_checkout_prices, api_client, checkout_with_shipping_address_for_cc, warehouses_for_cc):
    if False:
        return 10
    checkout = checkout_with_shipping_address_for_cc
    checkout.price_expiration = timezone.now()
    checkout.save(update_fields=['price_expiration'])
    query = UPDATE_CHECKOUT_DELIVERY_METHOD
    variables = {'token': checkout.token, 'deliveryMethodId': graphene.Node.to_global_id('Warehouse', warehouses_for_cc[1].pk)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    checkout.refresh_from_db()
    assert not response['data']['checkoutDeliveryMethodUpdate']['errors']
    assert mocked_invalidate_checkout_prices.call_count == 1

@freeze_time('2020-12-12 12:00:00')
def test_invalidate_checkout_prices_with_save(checkout, plugins_manager):
    if False:
        print('Hello World!')
    checkout.price_expiration = timezone.now() + timedelta(minutes=5)
    checkout.save(update_fields=['price_expiration'])
    (lines, _) = fetch_checkout_lines(checkout)
    checkout_info = fetch_checkout_info(checkout, lines, plugins_manager)
    updated_fields = invalidate_checkout_prices(checkout_info, lines, plugins_manager, save=True)
    checkout.refresh_from_db()
    assert checkout.price_expiration == timezone.now()
    assert updated_fields == ['price_expiration', 'last_change']

@freeze_time('2020-12-12 12:00:00')
def test_invalidate_checkout_prices_without_save(checkout, plugins_manager):
    if False:
        while True:
            i = 10
    original_expiration = checkout.price_expiration = timezone.now() + timedelta(minutes=5)
    checkout.save(update_fields=['price_expiration'])
    (lines, _) = fetch_checkout_lines(checkout)
    checkout_info = fetch_checkout_info(checkout, lines, plugins_manager)
    updated_fields = invalidate_checkout_prices(checkout_info, lines, plugins_manager, save=False)
    checkout.refresh_from_db()
    assert checkout.price_expiration == original_expiration
    assert updated_fields == ['price_expiration', 'last_change']