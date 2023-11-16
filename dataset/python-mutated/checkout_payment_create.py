from ...utils import get_graphql_content
CHECKOUT_PAYMENT_CREATE_MUTATION = '\nmutation createPayment($checkoutId: ID, $input: PaymentInput!) {\n  checkoutPaymentCreate(id: $checkoutId, input: $input) {\n    errors {\n      field\n      code\n      message\n    }\n    checkout {\n      id\n    }\n    payment {\n      id\n    }\n  }\n}\n'

def raw_checkout_dummy_payment_create(api_client, checkout_id, total_gross_amount, token):
    if False:
        return 10
    variables = {'checkoutId': checkout_id, 'input': {'amount': total_gross_amount, 'gateway': 'mirumee.payments.dummy', 'token': token}}
    response = api_client.post_graphql(CHECKOUT_PAYMENT_CREATE_MUTATION, variables=variables)
    content = get_graphql_content(response)
    checkout_data = content['data']['checkoutPaymentCreate']
    return checkout_data

def checkout_dummy_payment_create(api_client, checkout_id, total_gross_amount):
    if False:
        print('Hello World!')
    checkout_payment_create_response = raw_checkout_dummy_payment_create(api_client, checkout_id, total_gross_amount, token='fully_charged')
    assert checkout_payment_create_response['errors'] == []
    checkout_data = checkout_payment_create_response['checkout']
    assert checkout_data['id'] == checkout_id
    payment_data = checkout_payment_create_response['payment']
    assert payment_data['id'] is not None
    return payment_data