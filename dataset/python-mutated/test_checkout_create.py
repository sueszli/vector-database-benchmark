import graphene
from .....checkout.models import Checkout
from ....tests.utils import get_graphql_content

def test_checkout_create(api_client, stock, graphql_address_data, channel_USD):
    if False:
        return 10
    'Create checkout object using GraphQL API.'
    query = '\n        mutation createCheckout($checkoutInput: CheckoutCreateInput!) {\n        checkoutCreate(input: $checkoutInput) {\n            created\n            checkout {\n                id\n                token\n                email\n                quantity\n                shippingAddress {\n                    metadata {\n                        key\n                        value\n                    }\n                }\n                billingAddress {\n                    metadata {\n                        key\n                        value\n                    }\n                }\n            lines {\n                quantity\n            }\n            }\n            errors {\n                field\n                message\n                code\n                variants\n                addressType\n            }\n        }\n    }\n    '
    variant = stock.product_variant
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.id)
    test_email = 'test@example.com'
    address = graphql_address_data
    variables = {'checkoutInput': {'channel': channel_USD.slug, 'lines': [{'quantity': 1, 'variantId': variant_id}], 'email': test_email, 'shippingAddress': address, 'billingAddress': address}}
    assert not Checkout.objects.exists()
    response = api_client.post_graphql(query, variables)
    content = get_graphql_content(response)['data']['checkoutCreate']
    stored_metadata = {'public': 'public_value'}
    assert content['checkout']['shippingAddress']['metadata'] == graphql_address_data['metadata']
    assert content['checkout']['billingAddress']['metadata'] == graphql_address_data['metadata']
    checkout = Checkout.objects.get(email=test_email)
    assert checkout.billing_address.metadata == stored_metadata
    assert checkout.shipping_address.metadata == stored_metadata
    assert content['created'] is True