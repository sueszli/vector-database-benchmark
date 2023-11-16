from datetime import datetime, timedelta
import graphene
import pytz
from .....order import OrderStatus
from .....order.error_codes import OrderErrorCode
from .....product.models import ProductVariant
from ....tests.utils import get_graphql_content
ORDER_CAN_FINALIZE_QUERY = '\n    query OrderQuery($id: ID!){\n        order(id: $id){\n            canFinalize\n            errors {\n                code\n                field\n                message\n                warehouse\n                orderLines\n                variants\n            }\n        }\n    }\n'

def test_can_finalize_order(staff_api_client, permission_manage_orders, draft_order):
    if False:
        i = 10
        return i + 15
    order_id = graphene.Node.to_global_id('Order', draft_order.id)
    variables = {'id': order_id}
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    response = staff_api_client.post_graphql(ORDER_CAN_FINALIZE_QUERY, variables)
    content = get_graphql_content(response)
    assert content['data']['order']['canFinalize'] is True
    assert not content['data']['order']['errors']

def test_can_finalize_order_without_sku(staff_api_client, permission_manage_orders, draft_order):
    if False:
        print('Hello World!')
    ProductVariant.objects.update(sku=None)
    draft_order.lines.update(product_sku=None)
    order_id = graphene.Node.to_global_id('Order', draft_order.id)
    variables = {'id': order_id}
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    response = staff_api_client.post_graphql(ORDER_CAN_FINALIZE_QUERY, variables)
    content = get_graphql_content(response)
    assert content['data']['order']['canFinalize'] is True
    assert not content['data']['order']['errors']

def test_can_finalize_order_invalid_shipping_method_set(staff_api_client, permission_manage_orders, draft_order):
    if False:
        while True:
            i = 10
    order_id = graphene.Node.to_global_id('Order', draft_order.id)
    draft_order.channel.shipping_zones.clear()
    variables = {'id': order_id}
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    response = staff_api_client.post_graphql(ORDER_CAN_FINALIZE_QUERY, variables)
    content = get_graphql_content(response)
    assert content['data']['order']['canFinalize'] is False
    errors = content['data']['order']['errors']
    assert len(errors) == 3
    assert {error['code'] for error in errors} == {OrderErrorCode.SHIPPING_METHOD_NOT_APPLICABLE.name, OrderErrorCode.INSUFFICIENT_STOCK.name}
    assert {error['field'] for error in errors} == {'shipping', 'lines'}

def test_can_finalize_order_no_order_lines(staff_api_client, permission_manage_orders, order):
    if False:
        i = 10
        return i + 15
    order.status = OrderStatus.DRAFT
    order.save(update_fields=['status'])
    order_id = graphene.Node.to_global_id('Order', order.id)
    variables = {'id': order_id}
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    response = staff_api_client.post_graphql(ORDER_CAN_FINALIZE_QUERY, variables)
    content = get_graphql_content(response)
    assert content['data']['order']['canFinalize'] is False
    errors = content['data']['order']['errors']
    assert len(errors) == 1
    assert errors[0]['code'] == OrderErrorCode.REQUIRED.name
    assert errors[0]['field'] == 'lines'

def test_can_finalize_order_product_unavailable_for_purchase(staff_api_client, permission_manage_orders, draft_order):
    if False:
        for i in range(10):
            print('nop')
    order = draft_order
    order.status = OrderStatus.DRAFT
    order.save(update_fields=['status'])
    line = order.lines.first()
    product = line.variant.product
    product.channel_listings.update(available_for_purchase_at=None)
    order_id = graphene.Node.to_global_id('Order', order.id)
    variables = {'id': order_id}
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    response = staff_api_client.post_graphql(ORDER_CAN_FINALIZE_QUERY, variables)
    content = get_graphql_content(response)
    assert content['data']['order']['canFinalize'] is False
    errors = content['data']['order']['errors']
    assert len(errors) == 1
    assert errors[0]['code'] == OrderErrorCode.PRODUCT_UNAVAILABLE_FOR_PURCHASE.name
    assert errors[0]['field'] == 'lines'
    assert errors[0]['orderLines'] == [graphene.Node.to_global_id('OrderLine', line.pk)]

def test_can_finalize_order_product_available_for_purchase_from_tomorrow(staff_api_client, permission_manage_orders, draft_order):
    if False:
        print('Hello World!')
    order = draft_order
    order.status = OrderStatus.DRAFT
    order.save(update_fields=['status'])
    line = order.lines.first()
    product = line.variant.product
    product.channel_listings.update(available_for_purchase_at=datetime.now(pytz.UTC) + timedelta(days=1))
    order_id = graphene.Node.to_global_id('Order', order.id)
    variables = {'id': order_id}
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    response = staff_api_client.post_graphql(ORDER_CAN_FINALIZE_QUERY, variables)
    content = get_graphql_content(response)
    assert content['data']['order']['canFinalize'] is False
    errors = content['data']['order']['errors']
    assert len(errors) == 1
    assert errors[0]['code'] == OrderErrorCode.PRODUCT_UNAVAILABLE_FOR_PURCHASE.name
    assert errors[0]['field'] == 'lines'
    assert errors[0]['orderLines'] == [graphene.Node.to_global_id('OrderLine', line.pk)]

def test_can_finalize_order_invalid_voucher(staff_api_client, permission_manage_orders, draft_order_with_voucher):
    if False:
        for i in range(10):
            print('nop')
    order = draft_order_with_voucher
    order.voucher.channel_listings.all().delete()
    order_id = graphene.Node.to_global_id('Order', order.id)
    variables = {'id': order_id}
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    response = staff_api_client.post_graphql(ORDER_CAN_FINALIZE_QUERY, variables)
    content = get_graphql_content(response)
    assert content['data']['order']['canFinalize'] is False
    errors = content['data']['order']['errors']
    assert len(errors) == 1
    assert errors[0]['code'] == OrderErrorCode.INVALID_VOUCHER.name
    assert errors[0]['field'] == 'voucher'