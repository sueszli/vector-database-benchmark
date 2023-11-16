from .....core import JobStatus
from ....tests.utils import assert_no_permission, get_graphql_content
ORDERS_WITH_INVOICES_QUERY = '\n    query OrdersQuery {\n        orders(first: 5) {\n            edges {\n                node {\n                    invoices {\n                        status\n                        externalUrl\n                        number\n                    }\n                }\n            }\n        }\n    }\n'

def test_order_query_invoices(user_api_client, permission_group_manage_orders, fulfilled_order):
    if False:
        for i in range(10):
            print('nop')
    permission_group_manage_orders.user_set.add(user_api_client.user)
    response = user_api_client.post_graphql(ORDERS_WITH_INVOICES_QUERY)
    content = get_graphql_content(response)
    edges = content['data']['orders']['edges']
    assert len(edges) == 1
    assert edges[0]['node']['invoices'] == [{'status': JobStatus.SUCCESS.upper(), 'externalUrl': 'http://www.example.com/invoice.pdf', 'number': '01/12/2020/TEST'}]

def test_order_query_invoices_staff_no_permission(staff_api_client):
    if False:
        for i in range(10):
            print('nop')
    response = staff_api_client.post_graphql(ORDERS_WITH_INVOICES_QUERY)
    assert_no_permission(response)

def test_order_query_invoices_customer_user(user_api_client):
    if False:
        i = 10
        return i + 15
    response = user_api_client.post_graphql(ORDERS_WITH_INVOICES_QUERY)
    assert_no_permission(response)

def test_order_query_invoices_anonymous_user(api_client):
    if False:
        print('Hello World!')
    response = api_client.post_graphql(ORDERS_WITH_INVOICES_QUERY)
    assert_no_permission(response)

def test_order_query_invoices_app(app_api_client, permission_manage_orders, fulfilled_order):
    if False:
        while True:
            i = 10
    app_api_client.app.permissions.add(permission_manage_orders)
    response = app_api_client.post_graphql(ORDERS_WITH_INVOICES_QUERY)
    content = get_graphql_content(response)
    edges = content['data']['orders']['edges']
    assert len(edges) == 1
    assert edges[0]['node']['invoices'] == [{'status': JobStatus.SUCCESS.upper(), 'externalUrl': 'http://www.example.com/invoice.pdf', 'number': '01/12/2020/TEST'}]

def test_order_query_invoices_customer_user_by_token(api_client, fulfilled_order):
    if False:
        print('Hello World!')
    query = '\n    query OrderByToken($token: UUID!) {\n        orderByToken(token: $token) {\n            invoices {\n                status\n                number\n                externalUrl\n            }\n        }\n    }\n    '
    response = api_client.post_graphql(query, {'token': fulfilled_order.id})
    content = get_graphql_content(response)
    data = content['data']['orderByToken']
    assert data['invoices'] == [{'status': JobStatus.SUCCESS.upper(), 'externalUrl': 'http://www.example.com/invoice.pdf', 'number': '01/12/2020/TEST'}]