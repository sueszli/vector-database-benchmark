from ...utils import get_graphql_content
CHECKOUT_REMOVE_PROMO_CODE_MUTATION = '\nmutation CheckoutRemovePromoCode($id: ID, $promoCode: String) {\n    checkoutRemovePromoCode(\n        id: $id\n        promoCode: $promoCode\n    ) {\n        errors {\n            message\n            field\n            code\n        }\n        checkout {\n            id\n            voucherCode\n            totalPrice {\n                gross {\n                    amount\n                    }\n                }\n        }\n    }\n}\n'

def checkout_remove_promo_code(api_client, checkout_id, voucher_code):
    if False:
        i = 10
        return i + 15
    variables = {'id': checkout_id, 'promoCode': voucher_code}
    response = api_client.post_graphql(CHECKOUT_REMOVE_PROMO_CODE_MUTATION, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['checkoutRemovePromoCode']
    assert data['errors'] == []
    return data