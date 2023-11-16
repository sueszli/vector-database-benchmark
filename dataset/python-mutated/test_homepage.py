import pytest
from ....core.enums import ReportingPeriod
from ....tests.utils import get_graphql_content

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_retrieve_product_list(api_client, category, categories_tree, count_queries):
    if False:
        return 10
    query = '\n        query ProductsList {\n          shop {\n            description\n            name\n          }\n          categories(level: 0, first: 4) {\n            edges {\n              node {\n                id\n                name\n                backgroundImage {\n                  url\n                }\n              }\n            }\n          }\n        }\n    '
    get_graphql_content(api_client.post_graphql(query))

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_report_product_sales(staff_api_client, order_with_lines, order_with_lines_channel_PLN, permission_manage_products, permission_manage_orders, channel_USD, count_queries):
    if False:
        return 10
    query = '\n        query TopProducts($period: ReportingPeriod!, $channel: String!) {\n          reportProductSales(period: $period, first: 20, channel: $channel) {\n            edges {\n              node {\n                revenue(period: $period) {\n                  gross {\n                    amount\n                  }\n                }\n                quantityOrdered\n                sku\n              }\n            }\n          }\n        }\n    '
    variables = {'period': ReportingPeriod.TODAY.name, 'channel': channel_USD.slug}
    permissions = [permission_manage_orders, permission_manage_products]
    response = staff_api_client.post_graphql(query, variables, permissions)
    get_graphql_content(response)