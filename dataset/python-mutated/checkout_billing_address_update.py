from ... import DEFAULT_ADDRESS
from ...utils import get_graphql_content
CHECKOUT_BILLING_ADDRESS_UPDATE_MUTATION = '\nmutation CheckoutBillingAddressUpdate(\n  $billingAddress: AddressInput!, $checkoutId: ID!\n) {\n  checkoutBillingAddressUpdate(billingAddress: $billingAddress, id: $checkoutId) {\n    errors {\n      field\n      code\n      message\n    }\n    checkout {\n      billingAddress {\n        firstName\n        lastName\n        companyName\n        streetAddress1\n        postalCode\n        country {\n          code\n        }\n        city\n        countryArea\n        phone\n      }\n    }\n  }\n}\n'

def checkout_billing_address_update(api_client, checkout_id, address=DEFAULT_ADDRESS):
    if False:
        return 10
    variables = {'checkoutId': checkout_id, 'billingAddress': address}
    response = api_client.post_graphql(CHECKOUT_BILLING_ADDRESS_UPDATE_MUTATION, variables=variables)
    content = get_graphql_content(response)
    assert content['data']['checkoutBillingAddressUpdate']['errors'] == []
    data = content['data']['checkoutBillingAddressUpdate']['checkout']
    assert data['billingAddress']['firstName'] == address['firstName']
    assert data['billingAddress']['lastName'] == address['lastName']
    assert data['billingAddress']['companyName'] == address['companyName']
    assert data['billingAddress']['streetAddress1'] == address['streetAddress1']
    assert data['billingAddress']['postalCode'] == address['postalCode']
    assert data['billingAddress']['country']['code'] == address['country']
    assert data['billingAddress']['city'] == address['city'].upper()
    assert data['billingAddress']['countryArea'] == address['countryArea']
    assert data['billingAddress']['phone'] == address['phone']
    return data