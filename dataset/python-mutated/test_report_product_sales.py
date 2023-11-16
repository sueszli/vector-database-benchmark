from decimal import Decimal
from ....core.enums import ReportingPeriod
from ....tests.utils import get_graphql_content
QUERY_REPORT_PRODUCT_SALES = '\nquery TopProducts($period: ReportingPeriod!, $channel: String!) {\n    reportProductSales(period: $period, first: 20, channel: $channel) {\n        edges {\n            node {\n                revenue(period: $period) {\n                    gross {\n                        amount\n                    }\n                }\n                quantityOrdered\n                sku\n            }\n        }\n    }\n}\n'

def test_report_product_sales(staff_api_client, order_with_lines, order_with_lines_channel_PLN, permission_manage_products, permission_manage_orders, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    order = order_with_lines
    variables = {'period': ReportingPeriod.TODAY.name, 'channel': channel_USD.slug}
    permissions = [permission_manage_orders, permission_manage_products]
    response = staff_api_client.post_graphql(QUERY_REPORT_PRODUCT_SALES, variables, permissions)
    content = get_graphql_content(response)
    edges = content['data']['reportProductSales']['edges']
    node_a = edges[0]['node']
    line_a = order.lines.get(product_sku=node_a['sku'])
    assert node_a['quantityOrdered'] == line_a.quantity
    amount = str(node_a['revenue']['gross']['amount'])
    assert Decimal(amount) == line_a.quantity * line_a.unit_price_gross_amount
    node_b = edges[1]['node']
    line_b = order.lines.get(product_sku=node_b['sku'])
    assert node_b['quantityOrdered'] == line_b.quantity
    amount = str(node_b['revenue']['gross']['amount'])
    assert Decimal(amount) == line_b.quantity * line_b.unit_price_gross_amount

def test_report_product_sales_channel_pln(staff_api_client, order_with_lines, order_with_lines_channel_PLN, permission_manage_products, permission_manage_orders, channel_PLN):
    if False:
        return 10
    order = order_with_lines_channel_PLN
    variables = {'period': ReportingPeriod.TODAY.name, 'channel': channel_PLN.slug}
    permissions = [permission_manage_orders, permission_manage_products]
    response = staff_api_client.post_graphql(QUERY_REPORT_PRODUCT_SALES, variables, permissions)
    content = get_graphql_content(response)
    edges = content['data']['reportProductSales']['edges']
    node_a = edges[0]['node']
    line_a = order.lines.get(product_sku=node_a['sku'])
    assert node_a['quantityOrdered'] == line_a.quantity
    amount = str(node_a['revenue']['gross']['amount'])
    assert Decimal(amount) == line_a.quantity * line_a.unit_price_gross_amount
    node_b = edges[1]['node']
    line_b = order.lines.get(product_sku=node_b['sku'])
    assert node_b['quantityOrdered'] == line_b.quantity
    amount = str(node_b['revenue']['gross']['amount'])
    assert Decimal(amount) == line_b.quantity * line_b.unit_price_gross_amount

def test_report_product_sales_not_existing_channel(staff_api_client, order_with_lines, order_with_lines_channel_PLN, permission_manage_products, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    variables = {'period': ReportingPeriod.TODAY.name, 'channel': 'not-existing'}
    permissions = [permission_manage_orders, permission_manage_products]
    response = staff_api_client.post_graphql(QUERY_REPORT_PRODUCT_SALES, variables, permissions)
    content = get_graphql_content(response)
    assert not content['data']['reportProductSales']['edges']