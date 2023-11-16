import pytest
from .....payment.models import Transaction
from ....tests.utils import get_graphql_content
PAYMENT_TRANSACTIONS_QUERY = '\nquery {\n  orders(first:100) {\n    edges {\n      node {\n        payments {\n          id\n          gateway\n          isActive\n          created\n          modified\n          token\n          checkout {\n            id\n          }\n          order {\n            id\n          }\n          customerIpAddress\n          actions\n          total {\n            amount\n          }\n          capturedAmount {\n            amount\n          }\n          transactions {\n            id\n          }\n          availableRefundAmount {\n            amount\n          }\n          availableCaptureAmount {\n            amount\n          }\n          creditCard {\n            brand\n          }\n        }\n      }\n    }\n  }\n}\n'

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_payment_transactions(staff_api_client, orders_for_benchmarks, permission_group_manage_orders, count_queries):
    if False:
        i = 10
        return i + 15
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    transactions_count = 0
    content = get_graphql_content(staff_api_client.post_graphql(PAYMENT_TRANSACTIONS_QUERY, check_no_permissions=False))
    edges = content['data']['orders']['edges']
    for edge in edges:
        for payment in edge['node']['payments']:
            transactions_count += len(payment['transactions'])
    assert transactions_count == Transaction.objects.count() > 1