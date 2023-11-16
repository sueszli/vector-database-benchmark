from ....tests.utils import assert_no_permission, get_graphql_content
from .utils import PRIVATE_KEY, PRIVATE_VALUE, PUBLIC_KEY, PUBLIC_VALUE
SHOP_PRIVATE_AND_PUBLIC_METADATA_QUERY = '\n    query shopMetadata($key: String!, $privateKey: String!) {\n        shop {\n            metadata {\n                key\n                value\n            }\n            metafield(key: $key)\n            metafields(keys: [$key])\n            privateMetadata {\n                key\n                value\n            }\n            privateMetafield(key: $privateKey)\n            privateMetafields(keys: [$privateKey])\n        }\n}\n'
SHOP_PUBLIC_METADATA_QUERY = '\n    query shopMetadata($key: String!) {\n        shop {\n            metadata {\n                key\n                value\n            }\n            metafield(key: $key)\n            metafields(keys: [$key])\n        }\n}\n'

def test_customer_user_has_no_permission_to_shop_private_metadata(user_api_client, site_settings):
    if False:
        i = 10
        return i + 15
    site_settings.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    site_settings.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    site_settings.save()
    response = user_api_client.post_graphql(SHOP_PRIVATE_AND_PUBLIC_METADATA_QUERY, variables={'key': PUBLIC_KEY, 'privateKey': PRIVATE_KEY})
    assert_no_permission(response)

def test_customer_user_has_access_to_shop_public_metadata(user_api_client, site_settings):
    if False:
        return 10
    site_settings.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    site_settings.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    site_settings.save()
    response = user_api_client.post_graphql(SHOP_PUBLIC_METADATA_QUERY, variables={'key': PUBLIC_KEY})
    content = get_graphql_content(response)
    data = content['data']['shop']
    assert data['metadata'][0]['key'] == PUBLIC_KEY
    assert data['metadata'][0]['value'] == PUBLIC_VALUE
    assert data['metafield'] == PUBLIC_VALUE
    assert data['metafields'] == {PUBLIC_KEY: PUBLIC_VALUE}

def test_shop_metadata_query_as_staff_user(staff_api_client, site_settings, permission_manage_settings):
    if False:
        print('Hello World!')
    site_settings.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    site_settings.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    site_settings.save()
    response = staff_api_client.post_graphql(SHOP_PRIVATE_AND_PUBLIC_METADATA_QUERY, permissions=[permission_manage_settings], variables={'key': PUBLIC_KEY, 'privateKey': PRIVATE_KEY})
    content = get_graphql_content(response)
    data = content['data']['shop']
    assert data['metadata'][0]['key'] == PUBLIC_KEY
    assert data['metadata'][0]['value'] == PUBLIC_VALUE
    assert data['metafield'] == PUBLIC_VALUE
    assert data['metafields'] == {PUBLIC_KEY: PUBLIC_VALUE}
    assert data['privateMetadata'][0]['key'] == PRIVATE_KEY
    assert data['privateMetadata'][0]['value'] == PRIVATE_VALUE
    assert data['privateMetafield'] == PRIVATE_VALUE
    assert data['privateMetafields'] == {PRIVATE_KEY: PRIVATE_VALUE}