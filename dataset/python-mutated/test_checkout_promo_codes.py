import graphene
from .....checkout.error_codes import CheckoutErrorCode
from ....tests.utils import get_graphql_content
MUTATION_CHECKOUT_ADD_PROMO_CODE = '\n    mutation($checkoutId: ID, $token: UUID, $promoCode: String!) {\n        checkoutAddPromoCode(\n            checkoutId: $checkoutId, token: $token, promoCode: $promoCode) {\n            errors {\n                field\n                message\n                code\n            }\n            checkout {\n                id\n                token\n                voucherCode\n                giftCards {\n                    code\n                }\n                giftCards {\n                    id\n                    last4CodeChars\n                }\n                totalPrice {\n                    gross {\n                        amount\n                    }\n                }\n            }\n        }\n    }\n'

def _mutate_checkout_add_promo_code(client, variables):
    if False:
        while True:
            i = 10
    response = client.post_graphql(MUTATION_CHECKOUT_ADD_PROMO_CODE, variables)
    content = get_graphql_content(response)
    return content['data']['checkoutAddPromoCode']

def test_checkout_add_voucher_code_by_id(api_client, checkout_with_item, voucher):
    if False:
        for i in range(10):
            print('nop')
    checkout_id = graphene.Node.to_global_id('Checkout', checkout_with_item.pk)
    variables = {'checkoutId': checkout_id, 'promoCode': voucher.code}
    data = _mutate_checkout_add_promo_code(api_client, variables)
    assert not data['errors']
    assert data['checkout']['id'] == checkout_id
    assert data['checkout']['voucherCode'] == voucher.code

def test_checkout_add_voucher_code_by_token(api_client, checkout_with_item, voucher):
    if False:
        print('Hello World!')
    checkout_id = graphene.Node.to_global_id('Checkout', checkout_with_item.pk)
    variables = {'token': checkout_with_item.token, 'promoCode': voucher.code}
    data = _mutate_checkout_add_promo_code(api_client, variables)
    assert not data['errors']
    assert data['checkout']['id'] == checkout_id
    assert data['checkout']['voucherCode'] == voucher.code

def test_checkout_add_voucher_code_neither_token_and_id_given(api_client, checkout_with_item, voucher):
    if False:
        print('Hello World!')
    variables = {'promoCode': voucher.code}
    data = _mutate_checkout_add_promo_code(api_client, variables)
    assert len(data['errors']) == 1
    assert not data['checkout']
    assert data['errors'][0]['code'] == CheckoutErrorCode.GRAPHQL_ERROR.name

def test_checkout_add_voucher_code_both_token_and_id_given(api_client, checkout_with_item, voucher):
    if False:
        i = 10
        return i + 15
    checkout_id = graphene.Node.to_global_id('Checkout', checkout_with_item.pk)
    variables = {'promoCode': voucher.code, 'checkoutId': checkout_id, 'token': checkout_with_item.token}
    data = _mutate_checkout_add_promo_code(api_client, variables)
    assert len(data['errors']) == 1
    assert not data['checkout']
    assert data['errors'][0]['code'] == CheckoutErrorCode.GRAPHQL_ERROR.name
MUTATION_CHECKOUT_REMOVE_PROMO_CODE = '\n    mutation($checkoutId: ID, $token: UUID, $promoCode: String!) {\n        checkoutRemovePromoCode(\n            checkoutId: $checkoutId, token: $token, promoCode: $promoCode) {\n            errors {\n                field\n                message\n                code\n            }\n            checkout {\n                id,\n                voucherCode\n                giftCards {\n                    id\n                    last4CodeChars\n                }\n            }\n        }\n    }\n'

def _mutate_checkout_remove_promo_code(client, variables):
    if False:
        for i in range(10):
            print('nop')
    response = client.post_graphql(MUTATION_CHECKOUT_REMOVE_PROMO_CODE, variables)
    content = get_graphql_content(response)
    return content['data']['checkoutRemovePromoCode']

def test_checkout_remove_voucher_code(api_client, checkout_with_voucher):
    if False:
        print('Hello World!')
    assert checkout_with_voucher.voucher_code is not None
    checkout_id = graphene.Node.to_global_id('Checkout', checkout_with_voucher.pk)
    variables = {'checkoutId': checkout_id, 'promoCode': checkout_with_voucher.voucher_code}
    data = _mutate_checkout_remove_promo_code(api_client, variables)
    checkout_with_voucher.refresh_from_db()
    assert not data['errors']
    assert data['checkout']['id'] == checkout_id
    assert data['checkout']['voucherCode'] is None
    assert checkout_with_voucher.voucher_code is None

def test_checkout_remove_voucher_code_neither_token_and_id_given(api_client, checkout_with_voucher):
    if False:
        return 10
    assert checkout_with_voucher.voucher_code is not None
    variables = {'promoCode': checkout_with_voucher.voucher_code}
    data = _mutate_checkout_remove_promo_code(api_client, variables)
    assert len(data['errors']) == 1
    assert not data['checkout']
    assert data['errors'][0]['code'] == CheckoutErrorCode.GRAPHQL_ERROR.name

def test_checkout_remove_voucher_code_both_token_and_id_given(api_client, checkout_with_voucher):
    if False:
        i = 10
        return i + 15
    assert checkout_with_voucher.voucher_code is not None
    checkout_id = graphene.Node.to_global_id('Checkout', checkout_with_voucher.pk)
    variables = {'checkoutId': checkout_id, 'token': checkout_with_voucher.token, 'promoCode': checkout_with_voucher.voucher_code}
    data = _mutate_checkout_remove_promo_code(api_client, variables)
    assert len(data['errors']) == 1
    assert not data['checkout']
    assert data['errors'][0]['code'] == CheckoutErrorCode.GRAPHQL_ERROR.name