from saleor.graphql.tests.utils import get_graphql_content
ORDER_CANCEL_MUTATION = '\nmutation OrderCancel($id: ID!) {\n  orderCancel(id: $id) {\n    errors {\n        message\n        field\n    }\n    order {\n        id\n        status\n        paymentStatus\n        isPaid\n        totalBalance { amount }\n        total {\n            gross {\n                amount\n            }\n        }\n    }\n  }\n}\n'

def order_cancel(api_client, id):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': id}
    response = api_client.post_graphql(ORDER_CANCEL_MUTATION, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['orderCancel']
    errors = data['errors']
    assert errors == []
    return data