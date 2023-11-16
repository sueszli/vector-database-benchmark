from unittest.mock import patch
import graphene
import pytest
from .....payment import ChargeStatus
from .....warehouse.models import Stock
from ....tests.utils import get_graphql_content

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
@patch('saleor.order.actions.gateway.refund')
def test_fulfillment_refund_products_order_lines(mocked_refund, staff_api_client, permission_group_manage_orders, order_with_lines, payment_dummy, count_queries):
    if False:
        i = 10
        return i + 15
    query = '\n        mutation OrderFulfillmentRefundProducts(\n            $order: ID!, $input: OrderRefundProductsInput!\n        ) {\n            orderFulfillmentRefundProducts(\n                order: $order,\n                input: $input\n            ) {\n                fulfillment{\n                    id\n                    status\n                    lines{\n                        id\n                        quantity\n                        orderLine{\n                            id\n                        }\n                    }\n                }\n                errors {\n                    field\n                    code\n                    message\n                    warehouse\n                    orderLines\n                }\n            }\n        }\n    '
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    payment_dummy.total = order_with_lines.total_gross_amount
    payment_dummy.captured_amount = payment_dummy.total
    payment_dummy.charge_status = ChargeStatus.FULLY_CHARGED
    payment_dummy.save()
    order_with_lines.payments.add(payment_dummy)
    line_to_refund = order_with_lines.lines.first()
    order_id = graphene.Node.to_global_id('Order', order_with_lines.pk)
    line_id = graphene.Node.to_global_id('OrderLine', line_to_refund.pk)
    variables = {'order': order_id, 'input': {'orderLines': [{'orderLineId': line_id, 'quantity': 2}]}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    assert content['data']['orderFulfillmentRefundProducts']['fulfillment']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
@patch('saleor.plugins.manager.PluginsManager.product_variant_back_in_stock')
@patch('saleor.order.actions.gateway.refund')
def test_fulfillment_return_products_order_lines(mocked_refund, back_in_stock_webhook_mock, staff_api_client, permission_group_manage_orders, order_with_lines, payment_dummy, count_queries):
    if False:
        i = 10
        return i + 15
    query = '\n        mutation OrderFulfillmentReturnProducts(\n        $order: ID!, $input: OrderReturnProductsInput!\n    ) {\n        orderFulfillmentReturnProducts(\n            order: $order,\n            input: $input\n        ) {\n            returnFulfillment{\n                id\n                status\n                lines{\n                    id\n                    quantity\n                    orderLine{\n                        id\n                    }\n                }\n            }\n            replaceFulfillment{\n                id\n                status\n                lines{\n                    id\n                    quantity\n                    orderLine{\n                        id\n                    }\n                }\n            }\n            order{\n                id\n                status\n            }\n            replaceOrder{\n                id\n                status\n            }\n            errors {\n                field\n                code\n                message\n                warehouse\n                orderLines\n            }\n        }\n    }\n    '
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    payment_dummy.total = order_with_lines.total_gross_amount
    payment_dummy.captured_amount = payment_dummy.total
    payment_dummy.charge_status = ChargeStatus.FULLY_CHARGED
    payment_dummy.save()
    order_with_lines.payments.add(payment_dummy)
    line_to_return = order_with_lines.lines.first()
    line_quantity_to_return = 2
    line_to_replace = order_with_lines.lines.last()
    line_quantity_to_replace = 1
    order_id = graphene.Node.to_global_id('Order', order_with_lines.pk)
    line_id = graphene.Node.to_global_id('OrderLine', line_to_return.pk)
    replace_line_id = graphene.Node.to_global_id('OrderLine', line_to_replace.pk)
    variables = {'order': order_id, 'input': {'refund': True, 'includeShippingCosts': True, 'orderLines': [{'orderLineId': line_id, 'quantity': line_quantity_to_return, 'replace': False}, {'orderLineId': replace_line_id, 'quantity': line_quantity_to_replace, 'replace': True}]}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['orderFulfillmentReturnProducts']
    assert data['returnFulfillment']
    assert data['replaceFulfillment']
    back_in_stock_webhook_mock.assert_called_once_with(Stock.objects.last())