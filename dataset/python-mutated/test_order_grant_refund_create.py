from decimal import Decimal
import graphene
import pytest
from .....core.prices import quantize_price
from .....order.utils import update_order_charge_data
from ....core.utils import to_global_id_or_none
from ....tests.utils import assert_no_permission, get_graphql_content
from ...enums import OrderChargeStatusEnum, OrderGrantRefundCreateErrorCode, OrderGrantRefundCreateLineErrorCode
ORDER_GRANT_REFUND_CREATE = '\nmutation OrderGrantRefundCreate(\n    $id: ID!, $input: OrderGrantRefundCreateInput!\n){\n  orderGrantRefundCreate(id: $id, input:$input){\n    grantedRefund{\n      id\n      createdAt\n      updatedAt\n      amount{\n        amount\n      }\n      reason\n      user{\n        id\n      }\n      app{\n        id\n      }\n      shippingCostsIncluded\n      lines{\n        id\n        orderLine{\n          id\n        }\n        quantity\n        reason\n      }\n    }\n    order{\n      id\n      chargeStatus\n      grantedRefunds{\n        id\n        amount{\n          amount\n        }\n        createdAt\n        updatedAt\n        reason\n        app{\n          id\n        }\n        user{\n          id\n        }\n        shippingCostsIncluded\n        lines{\n          id\n          orderLine{\n            id\n          }\n          quantity\n          reason\n        }\n      }\n    }\n    errors{\n      field\n      code\n      lines{\n        field\n        code\n        lineId\n        message\n      }\n    }\n  }\n}\n'

@pytest.mark.parametrize('reason', ['', 'Reason', None])
def test_grant_refund_by_user(reason, staff_api_client, permission_manage_orders, order):
    if False:
        while True:
            i = 10
    order_id = to_global_id_or_none(order)
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    amount = Decimal('10.00')
    variables = {'id': order_id, 'input': {'amount': amount, 'reason': reason}}
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert not errors
    assert order_id == data['order']['id']
    assert len(data['order']['grantedRefunds']) == 1
    granted_refund_from_db = order.granted_refunds.first()
    granted_refund_assigned_to_order = data['order']['grantedRefunds'][0]
    assert granted_refund_assigned_to_order == data['grantedRefund']
    assert granted_refund_assigned_to_order['amount']['amount'] == granted_refund_from_db.amount_value == amount
    reason = reason or ''
    assert granted_refund_assigned_to_order['reason'] == reason == granted_refund_from_db.reason
    assert granted_refund_assigned_to_order['user']['id'] == to_global_id_or_none(staff_api_client.user) == to_global_id_or_none(granted_refund_from_db.user)
    assert not granted_refund_assigned_to_order['app']

@pytest.mark.parametrize('reason', ['', 'Reason', None])
def test_grant_refund_by_app(reason, app_api_client, permission_manage_orders, order):
    if False:
        i = 10
        return i + 15
    order_id = to_global_id_or_none(order)
    app_api_client.app.permissions.set([permission_manage_orders])
    amount = Decimal('10.00')
    variables = {'id': order_id, 'input': {'amount': amount, 'reason': reason}}
    response = app_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert not errors
    assert order_id == data['order']['id']
    assert len(data['order']['grantedRefunds']) == 1
    granted_refund_from_db = order.granted_refunds.first()
    granted_refund = data['order']['grantedRefunds'][0]
    assert granted_refund == data['grantedRefund']
    assert granted_refund['amount']['amount'] == amount == granted_refund_from_db.amount_value
    reason = reason or ''
    assert granted_refund['reason'] == reason == granted_refund_from_db.reason
    assert not granted_refund['user']
    assert granted_refund['app']['id'] == to_global_id_or_none(app_api_client.app) == to_global_id_or_none(granted_refund_from_db.app)

def test_grant_refund_by_app_missing_permission(app_api_client, order):
    if False:
        for i in range(10):
            print('nop')
    order_id = to_global_id_or_none(order)
    amount = Decimal('10.00')
    reason = 'Granted refund reason.'
    variables = {'id': order_id, 'input': {'amount': amount, 'reason': reason}}
    response = app_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    assert_no_permission(response)

def test_grant_refund_by_user_missing_permission(staff_api_client, order):
    if False:
        return 10
    order_id = to_global_id_or_none(order)
    amount = Decimal('10.00')
    reason = 'Granted refund reason.'
    variables = {'id': order_id, 'input': {'amount': amount, 'reason': reason}}
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    assert_no_permission(response)

def test_grant_refund_incorrect_order_id(staff_api_client, permission_manage_orders):
    if False:
        while True:
            i = 10
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    amount = Decimal('10.00')
    reason = 'Granted refund reason.'
    variables = {'id': 'wrong-id', 'input': {'amount': amount, 'reason': reason}}
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert len(errors) == 1
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == OrderGrantRefundCreateErrorCode.GRAPHQL_ERROR.name

def test_grant_refund_with_only_include_grant_refund_for_shipping(staff_api_client, permission_manage_orders, order_with_lines):
    if False:
        for i in range(10):
            print('nop')
    order_id = to_global_id_or_none(order_with_lines)
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    variables = {'id': order_id, 'input': {'grantRefundForShipping': True}}
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert not errors
    assert order_id == data['order']['id']
    assert len(data['order']['grantedRefunds']) == 1
    granted_refund_from_db = order_with_lines.granted_refunds.first()
    order_granted_refund = data['order']['grantedRefunds'][0]
    assert granted_refund_from_db.shipping_costs_included == order_granted_refund['shippingCostsIncluded'] is True
    assert data['grantedRefund']['shippingCostsIncluded'] is True
    assert granted_refund_from_db.amount_value == order_with_lines.shipping_price_gross_amount

def test_grant_refund_with_only_lines(staff_api_client, permission_manage_orders, order_with_lines):
    if False:
        return 10
    order = order_with_lines
    order_id = to_global_id_or_none(order)
    first_line = order.lines.first()
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    expected_reason = 'Reason'
    variables = {'id': order_id, 'input': {'lines': [{'id': to_global_id_or_none(first_line), 'quantity': 1, 'reason': expected_reason}]}}
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert not errors
    assert order_id == data['order']['id']
    assert len(data['order']['grantedRefunds']) == 1
    granted_refund_from_db = order.granted_refunds.first()
    order_granted_refund = data['order']['grantedRefunds'][0]
    assert data['grantedRefund']['shippingCostsIncluded'] is False
    assert len(order_granted_refund['lines']) == 1
    assert order_granted_refund['lines'][0]['quantity'] == 1
    assert order_granted_refund['lines'][0]['orderLine']['id'] == to_global_id_or_none(first_line)
    assert order_granted_refund['lines'][0]['reason'] == expected_reason
    assert granted_refund_from_db.amount_value == first_line.unit_price_gross_amount * 1
    assert quantize_price(granted_refund_from_db.amount_value, order.currency) == quantize_price(Decimal(order_granted_refund['amount']['amount']), order.currency)
    granted_refund_line = granted_refund_from_db.lines.first()
    assert granted_refund_line.order_line == first_line
    assert granted_refund_line.quantity == 1
    assert granted_refund_line.reason == expected_reason

def test_grant_refund_with_include_grant_refund_for_shipping_and_lines(staff_api_client, permission_manage_orders, order_with_lines):
    if False:
        return 10
    order = order_with_lines
    order_id = to_global_id_or_none(order)
    first_line = order.lines.first()
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    variables = {'id': order_id, 'input': {'grantRefundForShipping': True, 'lines': [{'id': to_global_id_or_none(first_line), 'quantity': 1}]}}
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert not errors
    assert order_id == data['order']['id']
    assert len(data['order']['grantedRefunds']) == 1
    granted_refund_from_db = order.granted_refunds.first()
    order_granted_refund = data['order']['grantedRefunds'][0]
    assert granted_refund_from_db.shipping_costs_included == order_granted_refund['shippingCostsIncluded'] is True
    assert data['grantedRefund']['shippingCostsIncluded'] is True
    assert len(order_granted_refund['lines']) == 1
    assert order_granted_refund['lines'][0]['quantity'] == 1
    assert order_granted_refund['lines'][0]['orderLine']['id'] == to_global_id_or_none(first_line)
    assert granted_refund_from_db.amount_value == first_line.unit_price_gross_amount * 1 + order.shipping_price_gross_amount
    assert quantize_price(granted_refund_from_db.amount_value, order.currency) == quantize_price(Decimal(order_granted_refund['amount']['amount']), order.currency)
    granted_refund_line = granted_refund_from_db.lines.first()
    assert granted_refund_line.order_line == first_line
    assert granted_refund_line.quantity == 1

def test_grant_refund_with_provided_lines_shipping_and_amount(staff_api_client, permission_manage_orders, order_with_lines):
    if False:
        while True:
            i = 10
    order = order_with_lines
    order_id = to_global_id_or_none(order)
    first_line = order.lines.first()
    expected_amount = Decimal('10.0')
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    variables = {'id': order_id, 'input': {'grantRefundForShipping': True, 'lines': [{'id': to_global_id_or_none(first_line), 'quantity': 1}], 'amount': expected_amount}}
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert not errors
    assert order_id == data['order']['id']
    assert len(data['order']['grantedRefunds']) == 1
    granted_refund_from_db = order.granted_refunds.first()
    order_granted_refund = data['order']['grantedRefunds'][0]
    assert granted_refund_from_db.shipping_costs_included == order_granted_refund['shippingCostsIncluded'] is True
    assert data['grantedRefund']['shippingCostsIncluded'] is True
    assert len(order_granted_refund['lines']) == 1
    assert order_granted_refund['lines'][0]['quantity'] == 1
    assert order_granted_refund['lines'][0]['orderLine']['id'] == to_global_id_or_none(first_line)
    assert granted_refund_from_db.amount_value == expected_amount
    assert quantize_price(granted_refund_from_db.amount_value, order.currency) == quantize_price(Decimal(order_granted_refund['amount']['amount']), order.currency)
    granted_refund_line = granted_refund_from_db.lines.first()
    assert granted_refund_line.order_line == first_line
    assert granted_refund_line.quantity == 1

def test_grant_refund_without_lines_and_amount_and_grant_for_shipping(staff_api_client, permission_manage_orders, order_with_lines):
    if False:
        i = 10
        return i + 15
    order = order_with_lines
    order_id = to_global_id_or_none(order)
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    variables = {'id': order_id, 'input': {'reason': 'Reason'}}
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert len(errors) == 3
    assert set([error['field'] for error in errors]) == {'amount', 'lines', 'grantRefundForShipping'}
    assert set([error['code'] for error in errors]) == {'REQUIRED'}

def test_grant_refund_with_incorrect_line_id(staff_api_client, permission_manage_orders, order_with_lines):
    if False:
        print('Hello World!')
    order = order_with_lines
    order_id = to_global_id_or_none(order)
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    variables = {'id': order_id, 'input': {'lines': [{'id': graphene.Node.to_global_id('OrderLine', 1), 'quantity': 1}]}}
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert len(errors) == 1
    error = errors[0]
    assert error['field'] == 'lines'
    assert error['code'] == OrderGrantRefundCreateErrorCode.INVALID.name
    assert len(error['lines']) == 1
    line = error['lines'][0]
    assert line['lineId'] == graphene.Node.to_global_id('OrderLine', 1)
    assert line['field'] == 'id'
    assert line['code'] == OrderGrantRefundCreateLineErrorCode.GRAPHQL_ERROR.name

def test_grant_refund_with_line_that_belongs_to_another_order(staff_api_client, permission_manage_orders, order_with_lines, order_with_lines_for_cc):
    if False:
        while True:
            i = 10
    order = order_with_lines
    another_order = order_with_lines_for_cc
    another_order_id = to_global_id_or_none(another_order)
    first_line = order.lines.first()
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    variables = {'id': another_order_id, 'input': {'lines': [{'id': to_global_id_or_none(first_line), 'quantity': 1}]}}
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert len(errors) == 1
    error = errors[0]
    assert error['field'] == 'lines'
    assert error['code'] == OrderGrantRefundCreateErrorCode.INVALID.name
    assert len(error['lines']) == 1
    line = error['lines'][0]
    assert line['lineId'] == to_global_id_or_none(first_line)
    assert line['field'] == 'id'
    assert line['code'] == OrderGrantRefundCreateLineErrorCode.NOT_FOUND.name

def test_grant_refund_with_bigger_quantity_than_available(staff_api_client, permission_manage_orders, order_with_lines):
    if False:
        for i in range(10):
            print('nop')
    order = order_with_lines
    order_id = to_global_id_or_none(order)
    first_line = order.lines.first()
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    variables = {'id': order_id, 'input': {'lines': [{'id': to_global_id_or_none(first_line), 'quantity': 100}]}}
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert len(errors) == 1
    error = errors[0]
    assert error['field'] == 'lines'
    assert error['code'] == OrderGrantRefundCreateErrorCode.INVALID.name
    assert len(error['lines']) == 1
    line = error['lines'][0]
    assert line['lineId'] == to_global_id_or_none(first_line)
    assert line['field'] == 'quantity'
    assert line['code'] == OrderGrantRefundCreateLineErrorCode.QUANTITY_GREATER_THAN_AVAILABLE.name

def test_grant_refund_with_refund_for_shipping_already_processed(staff_api_client, permission_manage_orders, order_with_lines):
    if False:
        return 10
    order = order_with_lines
    order_id = to_global_id_or_none(order)
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    variables = {'id': order_id, 'input': {'grantRefundForShipping': True}}
    order.granted_refunds.create(shipping_costs_included=True)
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert len(errors) == 1
    error = errors[0]
    assert error['field'] == 'grantRefundForShipping'
    assert error['code'] == OrderGrantRefundCreateErrorCode.SHIPPING_COSTS_ALREADY_GRANTED.name

def test_grant_refund_with_lines_and_existing_other_grant_refund(staff_api_client, permission_manage_orders, order_with_lines):
    if False:
        for i in range(10):
            print('nop')
    order = order_with_lines
    order_id = to_global_id_or_none(order)
    first_line = order.lines.first()
    first_line.quantity = 2
    first_line.save(update_fields=['quantity'])
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    variables = {'id': order_id, 'input': {'lines': [{'id': to_global_id_or_none(first_line), 'quantity': 1}]}}
    granted_refund = order.granted_refunds.create(shipping_costs_included=False)
    granted_refund.lines.create(order_line=first_line, quantity=1)
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert not errors
    assert order_id == data['order']['id']
    assert len(data['order']['grantedRefunds']) == 2
    granted_refund_from_db = order.granted_refunds.last()
    order_granted_refund = data['order']['grantedRefunds'][1]
    assert data['grantedRefund']['shippingCostsIncluded'] is False
    assert len(order_granted_refund['lines']) == 1
    assert order_granted_refund['lines'][0]['quantity'] == 1
    assert order_granted_refund['lines'][0]['orderLine']['id'] == to_global_id_or_none(first_line)
    assert granted_refund_from_db.amount_value == first_line.unit_price_gross_amount * 1
    assert quantize_price(granted_refund_from_db.amount_value, order.currency) == quantize_price(Decimal(order_granted_refund['amount']['amount']), order.currency)

def test_grant_refund_with_lines_and_existing_other_grant_and_refund_exceeding_quantity(staff_api_client, permission_manage_orders, order_with_lines):
    if False:
        print('Hello World!')
    order = order_with_lines
    order_id = to_global_id_or_none(order)
    first_line = order.lines.first()
    first_line.quantity = 1
    first_line.save(update_fields=['quantity'])
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    variables = {'id': order_id, 'input': {'lines': [{'id': to_global_id_or_none(first_line), 'quantity': 1}]}}
    granted_refund = order.granted_refunds.create(shipping_costs_included=False)
    granted_refund.lines.create(order_line=first_line, quantity=1)
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert len(errors) == 1
    error = errors[0]
    assert error['field'] == 'lines'
    assert error['code'] == OrderGrantRefundCreateErrorCode.INVALID.name
    assert len(error['lines']) == 1
    line = error['lines'][0]
    assert line['lineId'] == to_global_id_or_none(first_line)
    assert line['field'] == 'quantity'
    assert line['code'] == OrderGrantRefundCreateLineErrorCode.QUANTITY_GREATER_THAN_AVAILABLE.name

def test_grant_refund_updates_order_charge_status(staff_api_client, permission_manage_orders, order_with_lines):
    if False:
        while True:
            i = 10
    order = order_with_lines
    order_id = to_global_id_or_none(order)
    order.payment_transactions.create(charged_value=order.total.gross.amount, authorized_value=Decimal(12), currency=order_with_lines.currency)
    staff_api_client.user.user_permissions.add(permission_manage_orders)
    amount = Decimal('10.00')
    reason = 'Granted refund reason.'
    variables = {'id': order_id, 'input': {'amount': amount, 'reason': reason}}
    update_order_charge_data(order)
    assert order.charge_status == OrderChargeStatusEnum.FULL.value
    response = staff_api_client.post_graphql(ORDER_GRANT_REFUND_CREATE, variables)
    content = get_graphql_content(response)
    data = content['data']['orderGrantRefundCreate']
    errors = data['errors']
    assert not errors
    assert order_id == data['order']['id']
    assert len(data['order']['grantedRefunds']) == 1
    assert data['order']['chargeStatus'] == OrderChargeStatusEnum.OVERCHARGED.name