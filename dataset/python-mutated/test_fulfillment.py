import pytest
from ....tests.utils import get_graphql_content
FULFILLMENT_QUERY = '\nquery {\n    orders(first:100) {\n        edges {\n            node {\n                id\n                fulfillments {\n                    id\n                    status\n                    lines {\n                        id\n                        quantity\n                        orderLine {\n                            id\n                        }\n                    }\n                    warehouse {\n                        id\n                    }\n                }\n            }\n        }\n    }\n}\n'

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_fulfillment_query(staff_api_client, orders_for_benchmarks, permission_manage_orders, count_queries):
    if False:
        for i in range(10):
            print('nop')
    get_graphql_content(staff_api_client.post_graphql(FULFILLMENT_QUERY, permissions=[permission_manage_orders], check_no_permissions=False))