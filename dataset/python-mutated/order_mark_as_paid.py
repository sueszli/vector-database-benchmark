from saleor.graphql.tests.utils import get_graphql_content
ORDER_MARK_AS_PAID_MUTATION = '\nmutation OrderMarkAsPaid($id: ID!, $transactionReference: String) {\n  orderMarkAsPaid(id: $id, transactionReference: $transactionReference) {\n    order {\n        id\n        isPaid\n        paymentStatus\n        payments {\n            id\n            gateway\n            paymentMethodType\n        transactions {\n            kind\n        }\n      }\n        paymentStatusDisplay\n        status\n        statusDisplay\n        transactions {\n            id\n            order {\n            id\n            }\n            name\n        }\n    }\n    errors {\n        message\n        field\n    }\n  }\n}\n'

def mark_order_paid(api_client, id, transactionReference=None):
    if False:
        return 10
    variables = {'id': id, 'transactionReference': transactionReference}
    response = api_client.post_graphql(ORDER_MARK_AS_PAID_MUTATION, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['orderMarkAsPaid']
    errors = data['errors']
    assert errors == []
    return data