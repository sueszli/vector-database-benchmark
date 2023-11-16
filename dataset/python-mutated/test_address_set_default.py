import graphene
from ......checkout import AddressType
from .....tests.utils import get_graphql_content
SET_DEFAULT_ADDRESS_MUTATION = '\nmutation($address_id: ID!, $user_id: ID!, $type: AddressTypeEnum!) {\n  addressSetDefault(addressId: $address_id, userId: $user_id, type: $type) {\n    errors {\n      field\n      message\n    }\n    user {\n      defaultBillingAddress {\n        id\n      }\n      defaultShippingAddress {\n        id\n      }\n    }\n  }\n}\n'

def test_set_default_address(staff_api_client, address_other_country, customer_user, permission_manage_users):
    if False:
        for i in range(10):
            print('nop')
    customer_user.default_billing_address = None
    customer_user.default_shipping_address = None
    customer_user.save()
    address = address_other_country
    variables = {'address_id': graphene.Node.to_global_id('Address', address.id), 'user_id': graphene.Node.to_global_id('User', customer_user.id), 'type': AddressType.SHIPPING.upper()}
    response = staff_api_client.post_graphql(SET_DEFAULT_ADDRESS_MUTATION, variables, permissions=[permission_manage_users])
    content = get_graphql_content(response)
    data = content['data']['addressSetDefault']
    assert data['errors'][0]['field'] == 'addressId'
    address = customer_user.addresses.first()
    address_id = graphene.Node.to_global_id('Address', address.id)
    variables['address_id'] = address_id
    response = staff_api_client.post_graphql(SET_DEFAULT_ADDRESS_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['addressSetDefault']
    assert data['user']['defaultShippingAddress']['id'] == address_id