from saleor.graphql.tests.utils import get_graphql_content
ORDER_DISCOUNT_ADD_MUTATION = '\nmutation OrderDiscountAdd($input: OrderDiscountCommonInput!, $id: ID!) {\n  orderDiscountAdd(input:$input, orderId: $id) {\n    errors{\n        message\n        field\n        }\n    order {\n      errors{message field}\n      id\n      discounts {\n        id\n        value\n        valueType\n        type\n      }\n    }\n  }\n}\n'

def order_discount_add(api_client, id, input):
    if False:
        return 10
    variables = {'id': id, 'input': input}
    response = api_client.post_graphql(ORDER_DISCOUNT_ADD_MUTATION, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['orderDiscountAdd']
    order_id = data['order']['id']
    errors = data['errors']
    assert errors == []
    assert order_id is not None
    return data