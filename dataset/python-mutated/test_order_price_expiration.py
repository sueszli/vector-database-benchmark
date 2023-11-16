import pytest
from graphene import Node
from .....order import OrderStatus
from ....tests.utils import get_graphql_content
DRAFT_ORDER_UPDATE_MUTATION = '\nmutation draftUpdate(\n  $id: ID!\n  $voucher: ID!\n  $customerNote: String\n  $shippingAddress: AddressInput\n  $billingAddress: AddressInput\n) {\n  draftOrderUpdate(\n    id: $id\n    input: {\n      voucher: $voucher\n      customerNote: $customerNote\n      shippingAddress: $shippingAddress\n      billingAddress: $billingAddress\n    }\n  ) {\n    errors {\n      field\n      message\n    }\n  }\n}\n'

@pytest.fixture
def order_with_lines(order_with_lines):
    if False:
        while True:
            i = 10
    order_with_lines.status = OrderStatus.UNCONFIRMED
    order_with_lines.save(update_fields=['status'])
    return order_with_lines

def test_draft_order_update_shipping_address_invalidate_prices(staff_api_client, permission_group_manage_orders, draft_order, voucher, graphql_address_data):
    if False:
        i = 10
        return i + 15
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_MUTATION
    variables = {'id': Node.to_global_id('Order', draft_order.id), 'voucher': Node.to_global_id('Voucher', voucher.id), 'shippingAddress': graphql_address_data}
    content = get_graphql_content(staff_api_client.post_graphql(query, variables))
    assert not content['data']['draftOrderUpdate']['errors']
    draft_order.refresh_from_db()
    assert draft_order.should_refresh_prices

def test_draft_order_update_billing_address_invalidate_prices(staff_api_client, permission_group_manage_orders, draft_order, voucher, graphql_address_data):
    if False:
        i = 10
        return i + 15
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_MUTATION
    variables = {'id': Node.to_global_id('Order', draft_order.id), 'voucher': Node.to_global_id('Voucher', voucher.id), 'billingAddress': graphql_address_data}
    content = get_graphql_content(staff_api_client.post_graphql(query, variables))
    assert not content['data']['draftOrderUpdate']['errors']
    draft_order.refresh_from_db()
    assert draft_order.should_refresh_prices
ORDER_UPDATE_MUTATION = '\nmutation orderUpdate(\n  $id: ID!\n  $email: String\n  $shippingAddress: AddressInput\n  $billingAddress: AddressInput\n) {\n  orderUpdate(\n    id: $id\n    input: {\n      userEmail: $email\n      shippingAddress: $shippingAddress\n      billingAddress: $billingAddress\n    }\n  ) {\n    errors {\n      field\n      code\n    }\n    order {\n      userEmail\n    }\n  }\n}\n'

def test_order_update_shipping_address_invalidate_prices(staff_api_client, permission_group_manage_orders, order_with_lines, graphql_address_data):
    if False:
        while True:
            i = 10
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = order_with_lines
    order.user = None
    order.save()
    query = ORDER_UPDATE_MUTATION
    variables = {'id': Node.to_global_id('Order', order.id), 'shippingAddress': graphql_address_data}
    content = get_graphql_content(staff_api_client.post_graphql(query, variables))
    assert not content['data']['orderUpdate']['errors']
    order.refresh_from_db()
    assert order.should_refresh_prices

def test_order_update_billing_address_invalidate_prices(staff_api_client, permission_group_manage_orders, order_with_lines, graphql_address_data):
    if False:
        print('Hello World!')
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = order_with_lines
    order.user = None
    order.save()
    query = ORDER_UPDATE_MUTATION
    variables = {'id': Node.to_global_id('Order', order.id), 'billingAddress': graphql_address_data}
    content = get_graphql_content(staff_api_client.post_graphql(query, variables))
    assert not content['data']['orderUpdate']['errors']
    order.refresh_from_db()
    assert order.should_refresh_prices
ORDER_LINES_CREATE_MUTATION = '\nmutation OrderLinesCreate(\n  $orderId: ID!\n  $variantId: ID!\n  $quantity: Int!\n) {\n  orderLinesCreate(\n    id: $orderId\n    input: [\n      {\n        variantId: $variantId\n        quantity: $quantity\n      }\n    ]\n  ) {\n    errors {\n      field\n      message\n    }\n  }\n}\n'

def test_order_lines_create_invalidate_prices(order_with_lines, permission_group_manage_orders, staff_api_client):
    if False:
        while True:
            i = 10
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = ORDER_LINES_CREATE_MUTATION
    order = order_with_lines
    line = order.lines.first()
    variant = line.variant
    variables = {'orderId': Node.to_global_id('Order', order.id), 'variantId': Node.to_global_id('ProductVariant', variant.id), 'quantity': 2}
    content = get_graphql_content(staff_api_client.post_graphql(query, variables))
    assert not content['data']['orderLinesCreate']['errors']
    order.refresh_from_db()
    assert order.should_refresh_prices
ORDER_LINE_UPDATE_MUTATION = '\nmutation OrderLineUpdate(\n  $lineId: ID!\n  $quantity: Int!\n) {\n  orderLineUpdate(\n    id: $lineId\n    input: {\n      quantity: $quantity\n    }\n  ) {\n    errors {\n        field\n        message\n    }\n  }\n}\n'

def test_order_line_update_invalidate_prices(order_with_lines, permission_group_manage_orders, staff_api_client, staff_user):
    if False:
        while True:
            i = 10
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = ORDER_LINE_UPDATE_MUTATION
    order = order_with_lines
    line = order.lines.first()
    variables = {'lineId': Node.to_global_id('OrderLine', line.id), 'quantity': 1}
    content = get_graphql_content(staff_api_client.post_graphql(query, variables))
    assert not content['data']['orderLineUpdate']['errors']
    order.refresh_from_db()
    assert order.should_refresh_prices
ORDER_LINE_DELETE_MUTATION = '\nmutation OrderLineDelete(\n  $id: ID!\n) {\n  orderLineDelete(\n    id: $id\n  ) {\n    errors {\n      field\n      message\n    }\n  }\n}\n'

def test_order_line_remove(order_with_lines, permission_group_manage_orders, staff_api_client):
    if False:
        while True:
            i = 10
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = order_with_lines
    line = order.lines.first()
    query = ORDER_LINE_DELETE_MUTATION
    variables = {'id': Node.to_global_id('OrderLine', line.id)}
    content = get_graphql_content(staff_api_client.post_graphql(query, variables))
    assert not content['data']['orderLineDelete']['errors']
    order.refresh_from_db()
    assert order.should_refresh_prices