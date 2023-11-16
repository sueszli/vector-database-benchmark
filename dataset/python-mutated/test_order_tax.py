from decimal import Decimal
from functools import reduce
from operator import getitem
from unittest.mock import Mock, patch
import graphene
import pytest
from prices import Money, TaxedMoney
from .....order import OrderStatus
from .....order.interface import OrderTaxedPricesData
from ....tests.utils import get_graphql_content

@pytest.mark.parametrize(('fun_to_patch', 'price_name'), [('order_total', 'total'), ('order_undiscounted_total', 'undiscountedTotal'), ('order_shipping', 'shippingPrice')])
def test_order_resolver_tax_recalculation(staff_api_client, permission_manage_orders, order_with_lines, fun_to_patch, price_name):
    if False:
        print('Hello World!')
    price = TaxedMoney(net=Money(amount='1234.56', currency='USD'), gross=Money(amount='1267.89', currency='USD'))
    order = order_with_lines
    order.status = OrderStatus.UNCONFIRMED
    order.should_refresh_prices = True
    order.save()
    order_id = graphene.Node.to_global_id('Order', order.id)
    query = '\n        query OrderPrices($id: ID!) {\n            order(id: $id) {\n                %s { net { amount } gross { amount } }\n            }\n        }\n        ' % price_name
    variables = {'id': order_id}
    with patch(f'saleor.order.calculations.{fun_to_patch}', new=Mock(return_value=price)):
        response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_orders], check_no_permissions=False)
        content = get_graphql_content(response)
        data = content['data']['order']
    assert str(data[price_name]['net']['amount']) == str(price.net.amount)
    assert str(data[price_name]['gross']['amount']) == str(price.gross.amount)
ORDER_LINE_PRICE_DATA = OrderTaxedPricesData(price_with_discounts=TaxedMoney(net=Money(amount='1234.56', currency='USD'), gross=Money(amount='1267.89', currency='USD')), undiscounted_price=TaxedMoney(net=Money(amount='7234.56', currency='USD'), gross=Money(amount='7267.89', currency='USD')))

@pytest.mark.parametrize(('fun_to_patch', 'price_name', 'expected_price'), [('order_line_unit', 'unitPrice', ORDER_LINE_PRICE_DATA.price_with_discounts), ('order_line_unit', 'undiscountedUnitPrice', ORDER_LINE_PRICE_DATA.undiscounted_price), ('order_line_total', 'totalPrice', ORDER_LINE_PRICE_DATA.price_with_discounts)])
def test_order_line_resolver_tax_recalculation(staff_api_client, permission_manage_orders, order_with_lines, fun_to_patch, price_name, expected_price):
    if False:
        i = 10
        return i + 15
    order = order_with_lines
    order.status = OrderStatus.UNCONFIRMED
    order.should_refresh_prices = True
    order.save()
    order.lines.last().delete()
    order_id = graphene.Node.to_global_id('Order', order.id)
    query = '\n        query OrderLinePrices($id: ID!) {\n            order(id: $id) {\n                lines {\n                    %s { net { amount } gross { amount } }\n                }\n            }\n        }\n        ' % price_name
    variables = {'id': order_id}
    with patch(f'saleor.order.calculations.{fun_to_patch}', new=Mock(return_value=ORDER_LINE_PRICE_DATA)):
        response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_orders], check_no_permissions=False)
        content = get_graphql_content(response)
        data = content['data']['order']['lines'][0]
    assert str(data[price_name]['net']['amount']) == str(expected_price.net.amount)
    assert str(data[price_name]['gross']['amount']) == str(expected_price.gross.amount)
ORDER_SHIPPING_TAX_RATE_QUERY = '\nquery OrderShippingTaxRate($id: ID!) {\n    order(id: $id) {\n        shippingTaxRate\n    }\n}\n'
ORDER_LINE_TAX_RATE_QUERY = '\nquery OrderLineTaxRate($id: ID!) {\n    order(id: $id) {\n        lines {\n            taxRate\n        }\n    }\n}\n'

@pytest.mark.parametrize(('query', 'fun_to_patch', 'path'), [(ORDER_SHIPPING_TAX_RATE_QUERY, 'order_shipping_tax_rate', ['shippingTaxRate']), (ORDER_LINE_TAX_RATE_QUERY, 'order_line_tax_rate', ['lines', 0, 'taxRate'])])
def test_order_tax_rate_resolver_tax_recalculation(staff_api_client, permission_manage_orders, order_with_lines, query, fun_to_patch, path):
    if False:
        for i in range(10):
            print('nop')
    tax_rate = Decimal('0.01')
    order = order_with_lines
    order.status = OrderStatus.UNCONFIRMED
    order.should_refresh_prices = True
    order.save()
    order.lines.last().delete()
    order_id = graphene.Node.to_global_id('Order', order.id)
    variables = {'id': order_id}
    with patch(f'saleor.order.calculations.{fun_to_patch}', new=Mock(return_value=tax_rate)):
        response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_orders], check_no_permissions=False)
        content = get_graphql_content(response)
        data = content['data']['order']
    assert str(reduce(getitem, path, data)) == str(tax_rate)