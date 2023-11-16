from ... import DEFAULT_ADDRESS
from ...utils import get_graphql_content
CHECKOUT_CREATE_MUTATION = '\nmutation CreateCheckout($input: CheckoutCreateInput!) {\n  checkoutCreate(input: $input) {\n    errors {\n      field\n      code\n      message\n    }\n    checkout {\n      id\n      email\n      user {\n        id\n        email\n      }\n      channel {\n        slug\n      }\n      isShippingRequired\n      totalPrice {\n        gross {\n          amount\n        }\n        net {\n          amount\n        }\n        tax {\n          amount\n        }\n      }\n      created\n      isShippingRequired\n      shippingMethods {\n        id\n      }\n      deliveryMethod {\n        ... on ShippingMethod {\n          id\n        }\n        ... on Warehouse {\n          id\n        }\n      }\n      shippingMethod {\n        id\n      }\n      availableCollectionPoints {\n        id\n        isPrivate\n        clickAndCollectOption\n      }\n      lines {\n        id\n        totalPrice {\n          gross {\n            amount\n          }\n          net {\n            amount\n          }\n          tax {\n            amount\n          }\n        }\n        undiscountedTotalPrice {\n          amount\n        }\n        unitPrice {\n          gross {\n            amount\n          }\n        }\n        undiscountedUnitPrice {\n          amount\n        }\n        variant {\n          id\n        }\n      }\n    }\n  }\n}\n'

def raw_checkout_create(api_client, lines, channel_slug, email=None, set_default_billing_address=False, set_default_shipping_address=False):
    if False:
        return 10
    variables = {'input': {'channel': channel_slug, 'email': email, 'lines': lines}}
    if set_default_billing_address:
        variables['input']['billingAddress'] = DEFAULT_ADDRESS
    if set_default_shipping_address:
        variables['input']['shippingAddress'] = DEFAULT_ADDRESS
    response = api_client.post_graphql(CHECKOUT_CREATE_MUTATION, variables=variables)
    content = get_graphql_content(response)
    checkout_data = content['data']['checkoutCreate']
    return checkout_data

def checkout_create(api_client, lines, channel_slug, email=None, set_default_billing_address=False, set_default_shipping_address=False):
    if False:
        i = 10
        return i + 15
    checkout_response = raw_checkout_create(api_client, lines, channel_slug, email, set_default_billing_address, set_default_shipping_address)
    assert checkout_response['errors'] == []
    data = checkout_response['checkout']
    assert data['id'] is not None
    assert data['channel']['slug'] == channel_slug
    return data