from ... import DEFAULT_ADDRESS
from ...utils import get_graphql_content
CHECKOUT_SHIPPING_ADDRESS_UPDATE_MUTATION = '\nmutation CheckoutShippingAddressUpdate(\n  $shippingAddress: AddressInput!, $checkoutId: ID!\n) {\n  checkoutShippingAddressUpdate(\n    shippingAddress: $shippingAddress\n    id: $checkoutId\n  ) {\n    errors {\n      field\n      code\n      message\n    }\n    checkout {\n      shippingAddress {\n        firstName\n        lastName\n        companyName\n        streetAddress1\n        postalCode\n        country {\n          code\n        }\n        city\n        countryArea\n        phone\n      }\n      shippingMethods {\n        id\n        name\n      }\n    }\n  }\n}\n'

def checkout_shipping_address_update(api_client, checkout_id, address=DEFAULT_ADDRESS):
    if False:
        i = 10
        return i + 15
    variables = {'checkoutId': checkout_id, 'shippingAddress': address}
    response = api_client.post_graphql(CHECKOUT_SHIPPING_ADDRESS_UPDATE_MUTATION, variables=variables)
    content = get_graphql_content(response)
    assert content['data']['checkoutShippingAddressUpdate']['errors'] == []
    data = content['data']['checkoutShippingAddressUpdate']['checkout']
    assert data['shippingAddress']['firstName'] == address['firstName']
    assert data['shippingAddress']['lastName'] == address['lastName']
    assert data['shippingAddress']['companyName'] == address['companyName']
    assert data['shippingAddress']['streetAddress1'] == address['streetAddress1']
    assert data['shippingAddress']['postalCode'] == address['postalCode']
    assert data['shippingAddress']['country']['code'] == address['country']
    assert data['shippingAddress']['city'] == address['city'].upper()
    assert data['shippingAddress']['countryArea'] == address['countryArea']
    assert data['shippingAddress']['phone'] == address['phone']
    return data