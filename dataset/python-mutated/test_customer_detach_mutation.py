import graphene
from .....checkout.error_codes import CheckoutErrorCode
from ....tests.utils import get_graphql_content
MUTATION_CHECKOUT_CUSTOMER_DETACH = '\n    mutation checkoutCustomerDetach($checkoutId: ID, $token: UUID) {\n        checkoutCustomerDetach(checkoutId: $checkoutId, token: $token) {\n            checkout {\n                token\n            }\n            errors {\n                field\n                message\n                code\n            }\n        }\n    }\n    '

def test_checkout_customer_detach_by_id(user_api_client, checkout_with_item, customer_user):
    if False:
        return 10
    checkout = checkout_with_item
    checkout.user = customer_user
    checkout.save(update_fields=['user'])
    checkout_id = graphene.Node.to_global_id('Checkout', checkout.pk)
    variables = {'checkoutId': checkout_id}
    response = user_api_client.post_graphql(MUTATION_CHECKOUT_CUSTOMER_DETACH, variables)
    content = get_graphql_content(response)
    data = content['data']['checkoutCustomerDetach']
    assert not data['errors']
    checkout.refresh_from_db()
    assert checkout.user is None

def test_checkout_customer_detach_by_token(user_api_client, checkout_with_item, customer_user):
    if False:
        i = 10
        return i + 15
    checkout = checkout_with_item
    checkout.user = customer_user
    checkout.save(update_fields=['user'])
    variables = {'token': checkout.token}
    response = user_api_client.post_graphql(MUTATION_CHECKOUT_CUSTOMER_DETACH, variables)
    content = get_graphql_content(response)
    data = content['data']['checkoutCustomerDetach']
    assert not data['errors']
    checkout.refresh_from_db()
    assert checkout.user is None

def test_checkout_customer_detach_neither_token_and_id_given(user_api_client, checkout_with_item, customer_user):
    if False:
        i = 10
        return i + 15
    checkout = checkout_with_item
    checkout.user = customer_user
    checkout.save(update_fields=['user'])
    response = user_api_client.post_graphql(MUTATION_CHECKOUT_CUSTOMER_DETACH, {})
    content = get_graphql_content(response)
    data = content['data']['checkoutCustomerDetach']
    assert len(data['errors']) == 1
    assert not data['checkout']
    assert data['errors'][0]['code'] == CheckoutErrorCode.GRAPHQL_ERROR.name

def test_checkout_customer_detach_both_token_and_id_given(user_api_client, checkout_with_item, customer_user):
    if False:
        print('Hello World!')
    checkout = checkout_with_item
    checkout.user = customer_user
    checkout.save(update_fields=['user'])
    checkout_id = graphene.Node.to_global_id('Checkout', checkout.pk)
    variables = {'checkoutId': checkout_id, 'token': checkout.token}
    response = user_api_client.post_graphql(MUTATION_CHECKOUT_CUSTOMER_DETACH, variables)
    content = get_graphql_content(response)
    data = content['data']['checkoutCustomerDetach']
    assert len(data['errors']) == 1
    assert not data['checkout']
    assert data['errors'][0]['code'] == CheckoutErrorCode.GRAPHQL_ERROR.name