from unittest.mock import patch
import graphene
import pytest
from ....core.utils import to_global_id_or_none
from ....tests.utils import get_graphql_content
MUTATION_UPDATE_SHIPPING_METHOD = '\n    mutation checkoutShippingMethodUpdate(\n        $id: ID, $shippingMethodId: ID\n    ) {\n        checkoutShippingMethodUpdate(\n            id: $id, shippingMethodId: $shippingMethodId\n        ) {\n            errors {\n                field\n                message\n                code\n            }\n            checkout {\n                id\n                token\n                voucherCode\n                shippingMethod {\n                    id\n                }\n                shippingPrice {\n                    net {\n                        amount\n                    }\n                    gross {\n                        amount\n                    }\n                }\n            }\n        }\n    }\n'

@pytest.mark.django_db
@patch('saleor.graphql.checkout.mutations.checkout_shipping_method_update.clean_delivery_method')
def test_checkout_shipping_method_update_nullable_shipping_method_id(mock_clean_delivery_method, staff_api_client, shipping_method, checkout_with_item_and_voucher_and_shipping_method):
    if False:
        for i in range(10):
            print('nop')
    checkout = checkout_with_item_and_voucher_and_shipping_method
    mock_clean_delivery_method.return_value = True
    response = staff_api_client.post_graphql(MUTATION_UPDATE_SHIPPING_METHOD, variables={'id': to_global_id_or_none(checkout), 'shippingMethodId': None})
    data = get_graphql_content(response)['data']['checkoutShippingMethodUpdate']
    errors = data['errors']
    assert not errors
    assert data['checkout']['shippingMethod'] is None
    checkout.refresh_from_db(fields=['shipping_method'])
    assert checkout.shipping_method is None
    assert checkout.shipping_price.net.amount == 0
    assert checkout.shipping_price.gross.amount == 0
    assert checkout.voucher_code is not None
    response = staff_api_client.post_graphql(MUTATION_UPDATE_SHIPPING_METHOD, variables={'id': to_global_id_or_none(checkout), 'shippingMethodId': graphene.Node.to_global_id('ShippingMethod', shipping_method.pk)})
    data = get_graphql_content(response)['data']['checkoutShippingMethodUpdate']
    errors = data['errors']
    assert not errors
    checkout.refresh_from_db(fields=['shipping_method'])
    assert checkout.shipping_method is not None