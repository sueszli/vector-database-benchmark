from decimal import Decimal
from functools import partial
import graphene
import pytest
from prices import TaxedMoney, percentage_discount
from .....discount import DiscountValueType
from .....order import OrderEvents, OrderStatus
from ....discount.enums import DiscountValueTypeEnum
from ....tests.utils import get_graphql_content
ORDER_DISCOUNT_DELETE = '\nmutation OrderDiscountDelete($discountId: ID!){\n  orderDiscountDelete(discountId: $discountId){\n    order{\n      id\n    }\n    errors{\n      field\n      message\n      code\n    }\n  }\n}\n'

@pytest.mark.parametrize('status', [OrderStatus.DRAFT, OrderStatus.UNCONFIRMED])
def test_delete_order_discount_from_order_with_old_id(status, draft_order_with_fixed_discount_order, staff_api_client, permission_group_manage_orders):
    if False:
        print('Hello World!')
    order = draft_order_with_fixed_discount_order
    order.status = status
    order.save(update_fields=['status'])
    order_discount = draft_order_with_fixed_discount_order.discounts.get()
    name = 'discount translated'
    translated_name = 'discount translated name'
    order_discount.name = name
    order_discount.translated_name = translated_name
    order_discount.old_id = 1
    order_discount.save(update_fields=['name', 'translated_name', 'old_id'])
    current_undiscounted_total = order.undiscounted_total
    variables = {'discountId': graphene.Node.to_global_id('OrderDiscount', order_discount.old_id)}
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(ORDER_DISCOUNT_DELETE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderDiscountDelete']
    order.refresh_from_db()
    errors = data['errors']
    assert len(errors) == 0
    assert order.undiscounted_total == current_undiscounted_total
    assert order.total == current_undiscounted_total
    event = order.events.get()
    assert event.type == OrderEvents.ORDER_DISCOUNT_DELETED
    assert order.search_vector
ORDER_DISCOUNT_UPDATE = '\nmutation OrderDiscountUpdate($discountId: ID!, $input: OrderDiscountCommonInput!){\n  orderDiscountUpdate(discountId:$discountId, input: $input){\n    order{\n      id\n      total{\n        gross{\n          amount\n        }\n      }\n      undiscountedTotal{\n        gross{\n          amount\n        }\n      }\n    }\n    errors{\n        field\n        message\n        code\n      }\n  }\n}\n'

@pytest.mark.parametrize('status', [OrderStatus.DRAFT, OrderStatus.UNCONFIRMED])
def test_update_percentage_order_discount_by_old_id(status, draft_order_with_fixed_discount_order, staff_api_client, permission_group_manage_orders):
    if False:
        while True:
            i = 10
    order = draft_order_with_fixed_discount_order
    order.status = status
    order.save(update_fields=['status'])
    order_discount = draft_order_with_fixed_discount_order.discounts.get()
    order_discount.old_id = 1
    order_discount.save(update_fields=['old_id'])
    current_undiscounted_total = order.undiscounted_total
    reason = 'The reason of the discount'
    value = Decimal('10.000')
    variables = {'discountId': graphene.Node.to_global_id('OrderDiscount', order_discount.pk), 'input': {'valueType': DiscountValueTypeEnum.PERCENTAGE.name, 'value': value, 'reason': reason}}
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(ORDER_DISCOUNT_UPDATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderDiscountUpdate']
    order.refresh_from_db()
    discount = partial(percentage_discount, percentage=value)
    expected_net_total = discount(current_undiscounted_total.net)
    expected_gross_total = discount(current_undiscounted_total.gross)
    expected_total = TaxedMoney(expected_net_total, expected_gross_total)
    errors = data['errors']
    assert len(errors) == 0
    assert order.undiscounted_total.net == current_undiscounted_total.net
    assert expected_total.net == order.total.net
    assert order.discounts.count() == 1
    order_discount = order.discounts.first()
    assert order_discount.value == value
    assert order_discount.value_type == DiscountValueType.PERCENTAGE
    discount_amount = current_undiscounted_total.net - expected_total.net
    assert order_discount.amount == discount_amount
    assert order_discount.reason == reason
    event = order.events.get()
    assert event.type == OrderEvents.ORDER_DISCOUNT_UPDATED
    parameters = event.parameters
    discount_data = parameters.get('discount')
    assert discount_data['value'] == str(value)
    assert discount_data['value_type'] == DiscountValueTypeEnum.PERCENTAGE.value
    assert discount_data['amount_value'] == str(order_discount.amount.amount)