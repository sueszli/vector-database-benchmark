import graphene
from django.http import HttpResponse
from ....core.models import ModelWithMetadata
from ....order.models import Order
from ....payment.models import Payment
from ....payment.utils import payment_owned_by_user
from ....permission.models import Permission
from ...tests.fixtures import ApiClient
from ...tests.utils import assert_no_permission, get_graphql_content
PRIVATE_KEY = 'private_key'
PRIVATE_VALUE = 'private_vale'
PUBLIC_KEY = 'key'
PUBLIC_VALUE = 'value'

def execute_query(query_str: str, client: ApiClient, model: ModelWithMetadata, model_name: str, permissions: list[Permission]=None):
    if False:
        for i in range(10):
            print('nop')
    return client.post_graphql(query_str, variables={'id': graphene.Node.to_global_id(model_name, model.pk)}, permissions=[] if permissions is None else permissions, check_no_permissions=False)

def assert_model_contains_metadata(response: HttpResponse, model_name: str):
    if False:
        for i in range(10):
            print('nop')
    content = get_graphql_content(response)
    metadata = content['data'][model_name]['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def assert_model_contains_private_metadata(response: HttpResponse, model_name: str):
    if False:
        print('Hello World!')
    content = get_graphql_content(response)
    metadata = content['data'][model_name]['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_SELF_PUBLIC_META = '\n    {\n        me{\n            metadata{\n                key\n                value\n            }\n            metafields(keys: ["INVALID", "key"])\n            keyFieldValue: metafield(key: "key")\n        }\n    }\n'

def test_query_public_meta_for_me_as_customer(user_api_client):
    if False:
        print('Hello World!')
    me = user_api_client.user
    me.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    me.save(update_fields=['metadata'])
    response = user_api_client.post_graphql(QUERY_SELF_PUBLIC_META)
    content = get_graphql_content(response)
    metadata = content['data']['me']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
    metafields = content['data']['me']['metafields']
    assert metafields[PUBLIC_KEY] == PUBLIC_VALUE
    field_value = content['data']['me']['keyFieldValue']
    assert field_value == PUBLIC_VALUE

def test_query_public_meta_for_me_as_staff(staff_api_client):
    if False:
        return 10
    me = staff_api_client.user
    me.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    me.save(update_fields=['metadata'])
    response = staff_api_client.post_graphql(QUERY_SELF_PUBLIC_META)
    content = get_graphql_content(response)
    metadata = content['data']['me']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
    metafields = content['data']['me']['metafields']
    assert metafields[PUBLIC_KEY] == PUBLIC_VALUE
    field_value = content['data']['me']['keyFieldValue']
    assert field_value == PUBLIC_VALUE
QUERY_USER_PUBLIC_META = '\n    query userMeta($id: ID!){\n        user(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_customer_as_staff(staff_api_client, permission_manage_users, customer_user):
    if False:
        while True:
            i = 10
    customer_user.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    customer_user.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('User', customer_user.pk)}
    response = staff_api_client.post_graphql(QUERY_USER_PUBLIC_META, variables, [permission_manage_users])
    content = get_graphql_content(response)
    metadata = content['data']['user']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_customer_as_app(app_api_client, permission_manage_users, customer_user):
    if False:
        while True:
            i = 10
    customer_user.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    customer_user.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('User', customer_user.pk)}
    response = app_api_client.post_graphql(QUERY_USER_PUBLIC_META, variables, [permission_manage_users])
    content = get_graphql_content(response)
    metadata = content['data']['user']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_staff_as_other_staff(staff_api_client, permission_manage_staff, admin_user):
    if False:
        while True:
            i = 10
    admin_user.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    admin_user.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('User', admin_user.pk)}
    response = staff_api_client.post_graphql(QUERY_USER_PUBLIC_META, variables, [permission_manage_staff])
    content = get_graphql_content(response)
    metadata = content['data']['user']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_CHECKOUT_PUBLIC_META = '\n    query checkoutMeta($token: UUID!){\n        checkout(token: $token){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_checkout_as_anonymous_user(api_client, checkout):
    if False:
        i = 10
        return i + 15
    checkout.metadata_storage.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    checkout.metadata_storage.save(update_fields=['metadata'])
    variables = {'token': checkout.pk}
    response = api_client.post_graphql(QUERY_CHECKOUT_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['checkout']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_other_customer_checkout_as_anonymous_user(api_client, checkout, customer_user):
    if False:
        i = 10
        return i + 15
    checkout.user = customer_user
    checkout.metadata_storage.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    checkout.save(update_fields=['user'])
    checkout.metadata_storage.save(update_fields=['metadata'])
    variables = {'token': checkout.pk}
    response = api_client.post_graphql(QUERY_CHECKOUT_PUBLIC_META, variables)
    content = get_graphql_content(response)
    assert not content['data']['checkout']

def test_query_public_meta_for_checkout_as_customer(user_api_client, checkout):
    if False:
        i = 10
        return i + 15
    checkout.user = user_api_client.user
    checkout.metadata_storage.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    checkout.save(update_fields=['user'])
    checkout.metadata_storage.save(update_fields=['metadata'])
    variables = {'token': checkout.pk}
    response = user_api_client.post_graphql(QUERY_CHECKOUT_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['checkout']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_checkout_as_staff(staff_api_client, checkout, customer_user, permission_manage_checkouts):
    if False:
        i = 10
        return i + 15
    checkout.user = customer_user
    checkout.metadata_storage.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    checkout.save(update_fields=['user'])
    checkout.metadata_storage.save(update_fields=['metadata'])
    variables = {'token': checkout.pk}
    response = staff_api_client.post_graphql(QUERY_CHECKOUT_PUBLIC_META, variables, [permission_manage_checkouts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['checkout']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_checkout_as_app(app_api_client, checkout, customer_user, permission_manage_checkouts):
    if False:
        return 10
    checkout.user = customer_user
    checkout.metadata_storage.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    checkout.save(update_fields=['user'])
    checkout.metadata_storage.save(update_fields=['metadata'])
    variables = {'token': checkout.pk}
    response = app_api_client.post_graphql(QUERY_CHECKOUT_PUBLIC_META, variables, [permission_manage_checkouts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['checkout']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_ORDER_BY_TOKEN_PUBLIC_META = '\n    query orderMeta($token: UUID!){\n        orderByToken(token: $token){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_order_by_token_as_anonymous_user(api_client, order):
    if False:
        return 10
    order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    order.save(update_fields=['metadata'])
    variables = {'token': order.id}
    response = api_client.post_graphql(QUERY_ORDER_BY_TOKEN_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['orderByToken']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_order_by_token_as_customer(user_api_client, order):
    if False:
        i = 10
        return i + 15
    order.user = user_api_client.user
    order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    order.save(update_fields=['user', 'metadata'])
    variables = {'token': order.id}
    response = user_api_client.post_graphql(QUERY_ORDER_BY_TOKEN_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['orderByToken']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_order_by_token_as_staff(staff_api_client, order, customer_user, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    order.user = customer_user
    order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    order.save(update_fields=['user', 'metadata'])
    variables = {'token': order.id}
    response = staff_api_client.post_graphql(QUERY_ORDER_BY_TOKEN_PUBLIC_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['orderByToken']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_order_by_token_as_app(app_api_client, order, customer_user, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    order.user = customer_user
    order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    order.save(update_fields=['user', 'metadata'])
    variables = {'token': order.id}
    response = app_api_client.post_graphql(QUERY_ORDER_BY_TOKEN_PUBLIC_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['orderByToken']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_ORDER_PUBLIC_META = '\n    query orderMeta($id: ID!){\n        order(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_order_as_anonymous_user(api_client, order):
    if False:
        for i in range(10):
            print('nop')
    order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    order.save(update_fields=['user', 'metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', order.pk)}
    response = api_client.post_graphql(QUERY_ORDER_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['order']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_order_as_customer(user_api_client, order):
    if False:
        return 10
    order.user = user_api_client.user
    order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    order.save(update_fields=['user', 'metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', order.pk)}
    response = user_api_client.post_graphql(QUERY_ORDER_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['order']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_order_as_staff(staff_api_client, order, customer_user, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    order.user = customer_user
    order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    order.save(update_fields=['user', 'metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', order.pk)}
    response = staff_api_client.post_graphql(QUERY_ORDER_PUBLIC_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['order']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_order_as_app(app_api_client, order, customer_user, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    order.user = customer_user
    order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    order.save(update_fields=['user', 'metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', order.pk)}
    response = app_api_client.post_graphql(QUERY_ORDER_PUBLIC_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['order']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_DRAFT_ORDER_PUBLIC_META = '\n    query draftOrderMeta($id: ID!){\n        order(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_draft_order_as_anonymous_user(api_client, draft_order):
    if False:
        i = 10
        return i + 15
    draft_order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    draft_order.save(update_fields=['user', 'metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', draft_order.pk)}
    response = api_client.post_graphql(QUERY_DRAFT_ORDER_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['order']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_draft_order_as_customer(user_api_client, draft_order):
    if False:
        i = 10
        return i + 15
    draft_order.user = user_api_client.user
    draft_order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    draft_order.save(update_fields=['user', 'metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', draft_order.pk)}
    response = user_api_client.post_graphql(QUERY_DRAFT_ORDER_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['order']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_draft_order_as_staff(staff_api_client, draft_order, customer_user, permission_manage_orders):
    if False:
        print('Hello World!')
    draft_order.user = customer_user
    draft_order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    draft_order.save(update_fields=['user', 'metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', draft_order.pk)}
    response = staff_api_client.post_graphql(QUERY_DRAFT_ORDER_PUBLIC_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['order']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_draft_order_as_app(app_api_client, draft_order, customer_user, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    draft_order.user = customer_user
    draft_order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    draft_order.save(update_fields=['user', 'metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', draft_order.pk)}
    response = app_api_client.post_graphql(QUERY_ORDER_PUBLIC_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['order']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_FULFILLMENT_PUBLIC_META = '\n    query fulfillmentMeta($token: UUID!){\n        orderByToken(token: $token){\n            fulfillments{\n                metadata{\n                    key\n                    value\n                }\n          }\n        }\n    }\n'

def test_query_public_meta_for_fulfillment_as_anonymous_user(api_client, fulfilled_order):
    if False:
        for i in range(10):
            print('nop')
    fulfillment = fulfilled_order.fulfillments.first()
    fulfillment.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    fulfillment.save(update_fields=['metadata'])
    variables = {'token': fulfilled_order.id}
    response = api_client.post_graphql(QUERY_FULFILLMENT_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['orderByToken']['fulfillments'][0]['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_fulfillment_as_customer(user_api_client, fulfilled_order):
    if False:
        for i in range(10):
            print('nop')
    fulfillment = fulfilled_order.fulfillments.first()
    fulfillment.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    fulfillment.save(update_fields=['metadata'])
    fulfilled_order.user = user_api_client.user
    fulfilled_order.save(update_fields=['user'])
    variables = {'token': fulfilled_order.id}
    response = user_api_client.post_graphql(QUERY_FULFILLMENT_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['orderByToken']['fulfillments'][0]['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_fulfillment_as_staff(staff_api_client, fulfilled_order, customer_user, permission_manage_orders):
    if False:
        while True:
            i = 10
    fulfillment = fulfilled_order.fulfillments.first()
    fulfillment.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    fulfillment.save(update_fields=['metadata'])
    fulfilled_order.user = customer_user
    fulfilled_order.save(update_fields=['user'])
    variables = {'token': fulfilled_order.id}
    response = staff_api_client.post_graphql(QUERY_FULFILLMENT_PUBLIC_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['orderByToken']['fulfillments'][0]['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_fulfillment_as_app(app_api_client, fulfilled_order, customer_user, permission_manage_orders):
    if False:
        while True:
            i = 10
    fulfillment = fulfilled_order.fulfillments.first()
    fulfillment.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    fulfillment.save(update_fields=['metadata'])
    fulfilled_order.user = customer_user
    fulfilled_order.save(update_fields=['user'])
    variables = {'token': fulfilled_order.id}
    response = app_api_client.post_graphql(QUERY_FULFILLMENT_PUBLIC_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['orderByToken']['fulfillments'][0]['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_ATTRIBUTE_PUBLIC_META = '\n    query attributeMeta($id: ID!){\n        attribute(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_attribute_as_anonymous_user(api_client, color_attribute):
    if False:
        print('Hello World!')
    color_attribute.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    color_attribute.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Attribute', color_attribute.pk)}
    response = api_client.post_graphql(QUERY_ATTRIBUTE_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['attribute']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_attribute_as_customer(user_api_client, color_attribute):
    if False:
        for i in range(10):
            print('nop')
    color_attribute.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    color_attribute.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Attribute', color_attribute.pk)}
    response = user_api_client.post_graphql(QUERY_ATTRIBUTE_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['attribute']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_attribute_as_staff(staff_api_client, color_attribute, permission_manage_products):
    if False:
        i = 10
        return i + 15
    color_attribute.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    color_attribute.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Attribute', color_attribute.pk)}
    response = staff_api_client.post_graphql(QUERY_ATTRIBUTE_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['attribute']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_attribute_as_app(app_api_client, color_attribute, permission_manage_products):
    if False:
        for i in range(10):
            print('nop')
    color_attribute.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    color_attribute.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Attribute', color_attribute.pk)}
    response = app_api_client.post_graphql(QUERY_ATTRIBUTE_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['attribute']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_CATEGORY_PUBLIC_META = '\n    query categoryMeta($id: ID!){\n        category(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_category_as_anonymous_user(api_client, category):
    if False:
        while True:
            i = 10
    category.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    category.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Category', category.pk)}
    response = api_client.post_graphql(QUERY_CATEGORY_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['category']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_category_as_customer(user_api_client, category):
    if False:
        return 10
    category.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    category.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Category', category.pk)}
    response = user_api_client.post_graphql(QUERY_CATEGORY_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['category']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_category_as_staff(staff_api_client, category, permission_manage_products):
    if False:
        return 10
    category.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    category.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Category', category.pk)}
    response = staff_api_client.post_graphql(QUERY_CATEGORY_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['category']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_category_as_app(app_api_client, category, permission_manage_products):
    if False:
        while True:
            i = 10
    category.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    category.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Category', category.pk)}
    response = app_api_client.post_graphql(QUERY_CATEGORY_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['category']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_COLLECTION_PUBLIC_META = '\n    query collectionMeta($id: ID!, $channel: String) {\n        collection(id: $id, channel: $channel) {\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_collection_as_anonymous_user(api_client, published_collection, channel_USD):
    if False:
        print('Hello World!')
    collection = published_collection
    collection.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    collection.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Collection', collection.pk), 'channel': channel_USD.slug}
    response = api_client.post_graphql(QUERY_COLLECTION_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['collection']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_collection_as_customer(user_api_client, published_collection, channel_USD):
    if False:
        return 10
    collection = published_collection
    collection.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    collection.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Collection', collection.pk), 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(QUERY_COLLECTION_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['collection']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_collection_as_staff(staff_api_client, published_collection, permission_manage_products, channel_USD):
    if False:
        print('Hello World!')
    collection = published_collection
    collection.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    collection.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Collection', collection.pk), 'channel': channel_USD.slug}
    response = staff_api_client.post_graphql(QUERY_COLLECTION_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['collection']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_collection_as_app(app_api_client, published_collection, permission_manage_products, channel_USD):
    if False:
        print('Hello World!')
    collection = published_collection
    collection.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    collection.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Collection', collection.pk), 'channel': channel_USD.slug}
    response = app_api_client.post_graphql(QUERY_COLLECTION_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['collection']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_DIGITAL_CONTENT_PUBLIC_META = '\n    query digitalContentMeta($id: ID!){\n        digitalContent(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_digital_content_as_anonymous_user(api_client, digital_content):
    if False:
        i = 10
        return i + 15
    variables = {'id': graphene.Node.to_global_id('DigitalContent', digital_content.pk)}
    response = api_client.post_graphql(QUERY_DIGITAL_CONTENT_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_digital_content_as_customer(user_api_client, digital_content):
    if False:
        i = 10
        return i + 15
    digital_content.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    digital_content.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('DigitalContent', digital_content.pk)}
    response = user_api_client.post_graphql(QUERY_DIGITAL_CONTENT_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_digital_content_as_staff(staff_api_client, digital_content, permission_manage_products):
    if False:
        i = 10
        return i + 15
    digital_content.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    digital_content.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('DigitalContent', digital_content.pk)}
    response = staff_api_client.post_graphql(QUERY_DIGITAL_CONTENT_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['digitalContent']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_digital_content_as_app(app_api_client, digital_content, permission_manage_products):
    if False:
        return 10
    digital_content.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    digital_content.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('DigitalContent', digital_content.pk)}
    response = app_api_client.post_graphql(QUERY_DIGITAL_CONTENT_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['digitalContent']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_TRANSACTION_ITEM_PUBLIC_META = '\nquery transactionItemMeta($id: ID!){\n  order(id: $id){\n    transactions{\n      metadata{\n        key\n        value\n      }\n    }\n  }\n}\n'

def execute_query_public_metadata_for_transaction_item(client: ApiClient, order: Order, permissions: list[Permission]=None):
    if False:
        for i in range(10):
            print('nop')
    return execute_query(QUERY_TRANSACTION_ITEM_PUBLIC_META, client, order, 'Order', permissions)

def assert_transaction_item_contains_metadata(response):
    if False:
        print('Hello World!')
    content = get_graphql_content(response)
    metadata = content['data']['order']['transactions'][0]['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_transaction_item_as_customer(user_api_client, order, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    order.payment_transactions.create(metadata={PUBLIC_KEY: PUBLIC_VALUE})
    order.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    order.save(update_fields=['metadata'])
    response = execute_query_public_metadata_for_transaction_item(user_api_client, order, permissions=[])
    assert_no_permission(response)

def test_query_public_meta_for_transaction_item_as_staff_with_permission(staff_api_client, order_with_lines, permission_manage_orders, permission_manage_payments):
    if False:
        while True:
            i = 10
    order_with_lines.payment_transactions.create(metadata={PUBLIC_KEY: PUBLIC_VALUE})
    response = execute_query_public_metadata_for_transaction_item(staff_api_client, order_with_lines, permissions=[permission_manage_orders, permission_manage_payments])
    assert_transaction_item_contains_metadata(response)

def test_query_public_meta_for_transaction_item_as_staff_without_permission(staff_api_client, order):
    if False:
        i = 10
        return i + 15
    order.payment_transactions.create(metadata={PUBLIC_KEY: PUBLIC_VALUE})
    response = execute_query_public_metadata_for_transaction_item(staff_api_client, order)
    assert_no_permission(response)

def test_query_public_meta_for_transaction_item_as_app_with_permission(app_api_client, order, permission_manage_orders, permission_manage_payments):
    if False:
        i = 10
        return i + 15
    order.payment_transactions.create(metadata={PUBLIC_KEY: PUBLIC_VALUE})
    response = execute_query_public_metadata_for_transaction_item(app_api_client, order, permissions=[permission_manage_payments, permission_manage_orders])
    assert_transaction_item_contains_metadata(response)

def test_query_public_meta_for_transaction_item_as_app_without_permission(app_api_client, order):
    if False:
        return 10
    response = execute_query_public_metadata_for_transaction_item(app_api_client, order)
    assert_no_permission(response)
QUERY_PAYMENT_PUBLIC_META = '\n    query paymentsMeta($id: ID!){\n        payment(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def execute_query_public_metadata_for_payment(client: ApiClient, payment: Payment, permissions: list[Permission]=None):
    if False:
        for i in range(10):
            print('nop')
    return execute_query(QUERY_PAYMENT_PUBLIC_META, client, payment, 'Payment', permissions)

def assert_payment_contains_metadata(response):
    if False:
        for i in range(10):
            print('nop')
    assert_model_contains_metadata(response, 'payment')

def test_query_public_meta_for_payment_as_customer(user_api_client, payment_with_public_metadata, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    assert payment_owned_by_user(payment_with_public_metadata.pk, user_api_client.user)
    response = execute_query_public_metadata_for_payment(user_api_client, payment_with_public_metadata, permissions=[permission_manage_orders])
    assert_payment_contains_metadata(response)

def test_query_public_meta_for_payment_as_another_customer(user2_api_client, payment_with_public_metadata, permission_manage_orders):
    if False:
        print('Hello World!')
    assert not payment_owned_by_user(payment_with_public_metadata.pk, user2_api_client.user)
    response = execute_query_public_metadata_for_payment(user2_api_client, payment_with_public_metadata, permissions=[permission_manage_orders])
    assert_no_permission(response)

def test_query_public_meta_for_payment_as_staff_with_permission(staff_api_client, payment_with_public_metadata, permission_manage_orders, permission_manage_payments):
    if False:
        while True:
            i = 10
    response = execute_query_public_metadata_for_payment(staff_api_client, payment_with_public_metadata, permissions=[permission_manage_orders, permission_manage_payments])
    assert_payment_contains_metadata(response)

def test_query_public_meta_for_payment_as_staff_without_permission(staff_api_client, payment_with_public_metadata):
    if False:
        for i in range(10):
            print('nop')
    response = execute_query_public_metadata_for_payment(staff_api_client, payment_with_public_metadata)
    assert_no_permission(response)

def test_query_public_meta_for_payment_as_app_with_permission(app_api_client, payment_with_public_metadata, permission_manage_orders, permission_manage_payments):
    if False:
        print('Hello World!')
    response = execute_query_public_metadata_for_payment(app_api_client, payment_with_public_metadata, permissions=[permission_manage_orders, permission_manage_payments])
    assert_payment_contains_metadata(response)

def test_query_public_meta_for_payment_as_app_without_permission(app_api_client, payment_with_public_metadata):
    if False:
        return 10
    response = execute_query_public_metadata_for_payment(app_api_client, payment_with_public_metadata)
    assert_no_permission(response)
QUERY_PRODUCT_PUBLIC_META = '\n    query productsMeta($id: ID!, $channel: String){\n        product(id: $id, channel: $channel){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_product_as_anonymous_user(api_client, product, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    product.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    product.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Product', product.pk), 'channel': channel_USD.slug}
    response = api_client.post_graphql(QUERY_PRODUCT_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['product']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_product_as_customer(user_api_client, product, channel_USD):
    if False:
        while True:
            i = 10
    product.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    product.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Product', product.pk), 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(QUERY_PRODUCT_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['product']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_product_as_staff(staff_api_client, product, permission_manage_products):
    if False:
        while True:
            i = 10
    product.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    product.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Product', product.pk)}
    response = staff_api_client.post_graphql(QUERY_PRODUCT_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['product']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_product_as_app(app_api_client, product, permission_manage_products):
    if False:
        return 10
    product.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    product.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Product', product.pk)}
    response = app_api_client.post_graphql(QUERY_PRODUCT_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['product']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_PRODUCT_TYPE_PUBLIC_META = '\n    query productTypeMeta($id: ID!){\n        productType(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_product_type_as_anonymous_user(api_client, product_type):
    if False:
        while True:
            i = 10
    product_type.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    product_type.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductType', product_type.pk)}
    response = api_client.post_graphql(QUERY_PRODUCT_TYPE_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['productType']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_product_type_as_customer(user_api_client, product_type):
    if False:
        while True:
            i = 10
    product_type.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    product_type.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductType', product_type.pk)}
    response = user_api_client.post_graphql(QUERY_PRODUCT_TYPE_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['productType']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_product_type_as_staff(staff_api_client, product_type, permission_manage_products):
    if False:
        while True:
            i = 10
    product_type.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    product_type.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductType', product_type.pk)}
    response = staff_api_client.post_graphql(QUERY_PRODUCT_TYPE_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['productType']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_product_type_as_app(app_api_client, product_type, permission_manage_products):
    if False:
        i = 10
        return i + 15
    product_type.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    product_type.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductType', product_type.pk)}
    response = app_api_client.post_graphql(QUERY_PRODUCT_TYPE_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['productType']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_PRODUCT_VARIANT_PUBLIC_META = '\n    query productVariantMeta($id: ID!, $channel: String){\n        productVariant(id: $id, channel: $channel){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_product_variant_as_anonymous_user(api_client, variant, channel_USD):
    if False:
        return 10
    variant.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    variant.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductVariant', variant.pk), 'channel': channel_USD.slug}
    response = api_client.post_graphql(QUERY_PRODUCT_VARIANT_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['productVariant']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_product_variant_as_customer(user_api_client, variant, channel_USD):
    if False:
        return 10
    variant.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    variant.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductVariant', variant.pk), 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(QUERY_PRODUCT_VARIANT_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['productVariant']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_product_variant_as_staff(staff_api_client, variant, permission_manage_products):
    if False:
        return 10
    variant.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    variant.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductVariant', variant.pk)}
    response = staff_api_client.post_graphql(QUERY_PRODUCT_VARIANT_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['productVariant']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_product_variant_as_app(app_api_client, variant, permission_manage_products):
    if False:
        print('Hello World!')
    variant.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    variant.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductVariant', variant.pk)}
    response = app_api_client.post_graphql(QUERY_PRODUCT_VARIANT_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['productVariant']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_APP_PUBLIC_META = '\n    query appMeta($id: ID!){\n        app(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_app_as_anonymous_user(api_client, app):
    if False:
        print('Hello World!')
    variables = {'id': graphene.Node.to_global_id('App', app.pk)}
    response = api_client.post_graphql(QUERY_APP_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_app_as_customer(user_api_client, app):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': graphene.Node.to_global_id('App', app.pk)}
    response = user_api_client.post_graphql(QUERY_APP_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_app_as_staff(staff_api_client, app, permission_manage_apps):
    if False:
        print('Hello World!')
    app.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    app.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('App', app.pk)}
    response = staff_api_client.post_graphql(QUERY_APP_PUBLIC_META, variables, [permission_manage_apps], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['app']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_app_as_app(app_api_client, app, permission_manage_apps):
    if False:
        return 10
    app.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    app.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('App', app.pk)}
    response = app_api_client.post_graphql(QUERY_APP_PUBLIC_META, variables, [permission_manage_apps], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['app']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_PAGE_TYPE_PUBLIC_META = '\n    query pageTypeMeta($id: ID!){\n        pageType(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_page_type_as_anonymous_user(api_client, page_type):
    if False:
        for i in range(10):
            print('nop')
    page_type.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    page_type.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = api_client.post_graphql(QUERY_PAGE_TYPE_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['pageType']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_page_type_as_customer(user_api_client, page_type):
    if False:
        for i in range(10):
            print('nop')
    page_type.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    page_type.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = user_api_client.post_graphql(QUERY_PAGE_TYPE_PUBLIC_META, variables)
    content = get_graphql_content(response)
    metadata = content['data']['pageType']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_page_type_as_staff(staff_api_client, page_type, permission_manage_products):
    if False:
        print('Hello World!')
    page_type.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    page_type.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = staff_api_client.post_graphql(QUERY_PAGE_TYPE_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['pageType']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_page_type_as_app(app_api_client, page_type, permission_manage_products):
    if False:
        return 10
    page_type.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    page_type.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = app_api_client.post_graphql(QUERY_PAGE_TYPE_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['pageType']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_GIFT_CARD_PUBLIC_META = '\n    query giftCardMeta($id: ID!){\n        giftCard(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_gift_card_as_anonymous_user(api_client, gift_card):
    if False:
        return 10
    gift_card.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    gift_card.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = api_client.post_graphql(QUERY_GIFT_CARD_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_gift_card_as_customer(user_api_client, gift_card):
    if False:
        i = 10
        return i + 15
    gift_card.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    gift_card.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = user_api_client.post_graphql(QUERY_GIFT_CARD_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_gift_card_as_staff(staff_api_client, gift_card, permission_manage_gift_card):
    if False:
        while True:
            i = 10
    gift_card.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    gift_card.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = staff_api_client.post_graphql(QUERY_GIFT_CARD_PUBLIC_META, variables, [permission_manage_gift_card], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['giftCard']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_gift_card_as_app(app_api_client, gift_card, permission_manage_gift_card):
    if False:
        return 10
    gift_card.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    gift_card.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = app_api_client.post_graphql(QUERY_GIFT_CARD_PUBLIC_META, variables, [permission_manage_gift_card], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['giftCard']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_SELF_PRIVATE_META = '\n    {\n        me{\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_me_as_customer(user_api_client):
    if False:
        return 10
    response = user_api_client.post_graphql(QUERY_SELF_PRIVATE_META)
    assert_no_permission(response)

def test_query_private_meta_for_me_as_staff_with_manage_customer(staff_api_client, permission_manage_users):
    if False:
        return 10
    response = staff_api_client.post_graphql(QUERY_SELF_PRIVATE_META, None, [permission_manage_users])
    assert_no_permission(response)

def test_query_private_meta_for_me_as_staff_with_manage_staff(staff_api_client, permission_manage_staff):
    if False:
        while True:
            i = 10
    me = staff_api_client.user
    me.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    me.save(update_fields=['private_metadata'])
    response = staff_api_client.post_graphql(QUERY_SELF_PRIVATE_META, None, [permission_manage_staff])
    content = get_graphql_content(response)
    metadata = content['data']['me']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_USER_PRIVATE_META = '\n    query userMeta($id: ID!){\n        user(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_customer_as_staff(staff_api_client, permission_manage_users, customer_user):
    if False:
        for i in range(10):
            print('nop')
    customer_user.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    customer_user.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('User', customer_user.pk)}
    response = staff_api_client.post_graphql(QUERY_USER_PRIVATE_META, variables, [permission_manage_users])
    content = get_graphql_content(response)
    metadata = content['data']['user']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_customer_as_app(app_api_client, permission_manage_users, customer_user):
    if False:
        return 10
    customer_user.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    customer_user.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('User', customer_user.pk)}
    response = app_api_client.post_graphql(QUERY_USER_PRIVATE_META, variables, [permission_manage_users])
    content = get_graphql_content(response)
    metadata = content['data']['user']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_staff_as_other_staff(staff_api_client, permission_manage_staff, admin_user):
    if False:
        return 10
    admin_user.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    admin_user.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('User', admin_user.pk)}
    response = staff_api_client.post_graphql(QUERY_USER_PRIVATE_META, variables, [permission_manage_staff])
    content = get_graphql_content(response)
    metadata = content['data']['user']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_CHECKOUT_PRIVATE_META = '\n    query checkoutMeta($token: UUID!){\n        checkout(token: $token){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_checkout_as_anonymous_user(api_client, checkout):
    if False:
        while True:
            i = 10
    variables = {'token': checkout.pk}
    response = api_client.post_graphql(QUERY_CHECKOUT_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_other_customer_checkout_as_anonymous_user(api_client, checkout, customer_user):
    if False:
        for i in range(10):
            print('nop')
    checkout.user = customer_user
    checkout.save(update_fields=['user'])
    variables = {'token': checkout.pk}
    response = api_client.post_graphql(QUERY_CHECKOUT_PRIVATE_META, variables)
    content = get_graphql_content(response)
    assert not content['data']['checkout']

def test_query_private_meta_for_checkout_as_customer(user_api_client, checkout):
    if False:
        i = 10
        return i + 15
    checkout.user = user_api_client.user
    checkout.save(update_fields=['user'])
    variables = {'token': checkout.pk}
    response = user_api_client.post_graphql(QUERY_CHECKOUT_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_checkout_as_staff(staff_api_client, checkout, customer_user, permission_manage_checkouts):
    if False:
        for i in range(10):
            print('nop')
    checkout.user = customer_user
    checkout.metadata_storage.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    checkout.save(update_fields=['user'])
    checkout.metadata_storage.save(update_fields=['private_metadata'])
    variables = {'token': checkout.pk}
    response = staff_api_client.post_graphql(QUERY_CHECKOUT_PRIVATE_META, variables, [permission_manage_checkouts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['checkout']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_checkout_as_app(app_api_client, checkout, customer_user, permission_manage_checkouts):
    if False:
        for i in range(10):
            print('nop')
    checkout.user = customer_user
    checkout.metadata_storage.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    checkout.save(update_fields=['user'])
    checkout.metadata_storage.save(update_fields=['private_metadata'])
    variables = {'token': checkout.pk}
    response = app_api_client.post_graphql(QUERY_CHECKOUT_PRIVATE_META, variables, [permission_manage_checkouts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['checkout']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_ORDER_BY_TOKEN_PRIVATE_META = '\n    query orderMeta($token: UUID!){\n        orderByToken(token: $token){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_order_by_token_as_anonymous_user(api_client, order):
    if False:
        return 10
    variables = {'token': order.id}
    response = api_client.post_graphql(QUERY_ORDER_BY_TOKEN_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_order_by_token_as_customer(user_api_client, order):
    if False:
        return 10
    order.user = user_api_client.user
    order.save(update_fields=['user'])
    variables = {'token': order.id}
    response = user_api_client.post_graphql(QUERY_ORDER_BY_TOKEN_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_order_by_token_as_staff(staff_api_client, order, customer_user, permission_manage_orders):
    if False:
        return 10
    order.user = customer_user
    order.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    order.save(update_fields=['user', 'private_metadata'])
    variables = {'token': order.id}
    response = staff_api_client.post_graphql(QUERY_ORDER_BY_TOKEN_PRIVATE_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['orderByToken']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_order_by_token_as_app(app_api_client, order, customer_user, permission_manage_orders):
    if False:
        print('Hello World!')
    order.user = customer_user
    order.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    order.save(update_fields=['user', 'private_metadata'])
    variables = {'token': order.id}
    response = app_api_client.post_graphql(QUERY_ORDER_BY_TOKEN_PRIVATE_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['orderByToken']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_ORDER_PRIVATE_META = '\n    query orderMeta($id: ID!){\n        order(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_order_as_anonymous_user(api_client, order):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': graphene.Node.to_global_id('Order', order.pk)}
    response = api_client.post_graphql(QUERY_ORDER_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_order_as_customer(user_api_client, order):
    if False:
        i = 10
        return i + 15
    order.user = user_api_client.user
    order.save(update_fields=['user'])
    variables = {'id': graphene.Node.to_global_id('Order', order.pk)}
    response = user_api_client.post_graphql(QUERY_ORDER_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_order_as_staff(staff_api_client, order, customer_user, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    order.user = customer_user
    order.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    order.save(update_fields=['user', 'private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', order.pk)}
    response = staff_api_client.post_graphql(QUERY_ORDER_PRIVATE_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['order']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_order_as_app(app_api_client, order, customer_user, permission_manage_orders):
    if False:
        print('Hello World!')
    order.user = customer_user
    order.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    order.save(update_fields=['user', 'private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', order.pk)}
    response = app_api_client.post_graphql(QUERY_ORDER_PRIVATE_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['order']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_DRAFT_ORDER_PRIVATE_META = '\n    query draftOrderMeta($id: ID!){\n        order(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_draft_order_as_anonymous_user(api_client, draft_order):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': graphene.Node.to_global_id('Order', draft_order.pk)}
    response = api_client.post_graphql(QUERY_DRAFT_ORDER_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_draft_order_as_customer(user_api_client, draft_order):
    if False:
        while True:
            i = 10
    draft_order.user = user_api_client.user
    draft_order.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    draft_order.save(update_fields=['user', 'private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', draft_order.pk)}
    response = user_api_client.post_graphql(QUERY_DRAFT_ORDER_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_draft_order_as_staff(staff_api_client, draft_order, customer_user, permission_manage_orders):
    if False:
        for i in range(10):
            print('nop')
    draft_order.user = customer_user
    draft_order.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    draft_order.save(update_fields=['user', 'private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', draft_order.pk)}
    response = staff_api_client.post_graphql(QUERY_DRAFT_ORDER_PRIVATE_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['order']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_draft_order_as_app(app_api_client, draft_order, customer_user, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    draft_order.user = customer_user
    draft_order.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    draft_order.save(update_fields=['user', 'private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Order', draft_order.pk)}
    response = app_api_client.post_graphql(QUERY_ORDER_PRIVATE_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['order']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_FULFILLMENT_PRIVATE_META = '\n    query fulfillmentMeta($token: UUID!){\n        orderByToken(token: $token){\n            fulfillments{\n                privateMetadata{\n                    key\n                    value\n                }\n          }\n        }\n    }\n'

def test_query_private_meta_for_fulfillment_as_anonymous_user(api_client, fulfilled_order):
    if False:
        print('Hello World!')
    variables = {'token': fulfilled_order.id}
    response = api_client.post_graphql(QUERY_FULFILLMENT_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_fulfillment_as_customer(user_api_client, fulfilled_order):
    if False:
        return 10
    fulfilled_order.user = user_api_client.user
    fulfilled_order.save(update_fields=['user'])
    variables = {'token': fulfilled_order.id}
    response = user_api_client.post_graphql(QUERY_FULFILLMENT_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_fulfillment_as_staff(staff_api_client, fulfilled_order, customer_user, permission_manage_orders):
    if False:
        while True:
            i = 10
    fulfillment = fulfilled_order.fulfillments.first()
    fulfillment.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    fulfillment.save(update_fields=['private_metadata'])
    fulfilled_order.user = customer_user
    fulfilled_order.save(update_fields=['user'])
    variables = {'token': fulfilled_order.id}
    response = staff_api_client.post_graphql(QUERY_FULFILLMENT_PRIVATE_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['orderByToken']['fulfillments'][0]['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_fulfillment_as_app(app_api_client, fulfilled_order, customer_user, permission_manage_orders):
    if False:
        return 10
    fulfillment = fulfilled_order.fulfillments.first()
    fulfillment.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    fulfillment.save(update_fields=['private_metadata'])
    fulfilled_order.user = customer_user
    fulfilled_order.save(update_fields=['user'])
    variables = {'token': fulfilled_order.id}
    response = app_api_client.post_graphql(QUERY_FULFILLMENT_PRIVATE_META, variables, [permission_manage_orders], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['orderByToken']['fulfillments'][0]['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_ATTRIBUTE_PRIVATE_META = '\n    query attributeMeta($id: ID!){\n        attribute(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_attribute_as_anonymous_user(api_client, color_attribute):
    if False:
        print('Hello World!')
    variables = {'id': graphene.Node.to_global_id('Attribute', color_attribute.pk)}
    response = api_client.post_graphql(QUERY_ATTRIBUTE_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_attribute_as_customer(user_api_client, color_attribute):
    if False:
        return 10
    variables = {'id': graphene.Node.to_global_id('Attribute', color_attribute.pk)}
    response = user_api_client.post_graphql(QUERY_ATTRIBUTE_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_attribute_as_staff(staff_api_client, color_attribute, permission_manage_product_types_and_attributes):
    if False:
        while True:
            i = 10
    color_attribute.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    color_attribute.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Attribute', color_attribute.pk)}
    response = staff_api_client.post_graphql(QUERY_ATTRIBUTE_PRIVATE_META, variables, [permission_manage_product_types_and_attributes], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['attribute']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_attribute_as_app(app_api_client, color_attribute, permission_manage_product_types_and_attributes):
    if False:
        return 10
    color_attribute.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    color_attribute.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Attribute', color_attribute.pk)}
    response = app_api_client.post_graphql(QUERY_ATTRIBUTE_PRIVATE_META, variables, [permission_manage_product_types_and_attributes], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['attribute']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_CATEGORY_PRIVATE_META = '\n    query categoryMeta($id: ID!){\n        category(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_category_as_anonymous_user(api_client, category):
    if False:
        i = 10
        return i + 15
    variables = {'id': graphene.Node.to_global_id('Category', category.pk)}
    response = api_client.post_graphql(QUERY_CATEGORY_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_category_as_customer(user_api_client, category):
    if False:
        i = 10
        return i + 15
    variables = {'id': graphene.Node.to_global_id('Category', category.pk)}
    response = user_api_client.post_graphql(QUERY_CATEGORY_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_category_as_staff(staff_api_client, category, permission_manage_products):
    if False:
        while True:
            i = 10
    category.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    category.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Category', category.pk)}
    response = staff_api_client.post_graphql(QUERY_CATEGORY_PRIVATE_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['category']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_category_as_app(app_api_client, category, permission_manage_products):
    if False:
        i = 10
        return i + 15
    category.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    category.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Category', category.pk)}
    response = app_api_client.post_graphql(QUERY_CATEGORY_PRIVATE_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['category']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_COLLECTION_PRIVATE_META = '\n    query collectionMeta($id: ID!, $channel: String){\n        collection(id: $id, channel: $channel){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_collection_as_anonymous_user(api_client, published_collection, channel_USD):
    if False:
        i = 10
        return i + 15
    variables = {'id': graphene.Node.to_global_id('Collection', published_collection.pk), 'channel': channel_USD.slug}
    response = api_client.post_graphql(QUERY_COLLECTION_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_collection_as_customer(user_api_client, published_collection, channel_USD):
    if False:
        print('Hello World!')
    variables = {'id': graphene.Node.to_global_id('Collection', published_collection.pk), 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(QUERY_COLLECTION_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_collection_as_staff(staff_api_client, published_collection, permission_manage_products, channel_USD):
    if False:
        return 10
    collection = published_collection
    collection.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    collection.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Collection', published_collection.pk), 'channel': channel_USD.slug}
    response = staff_api_client.post_graphql(QUERY_COLLECTION_PRIVATE_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['collection']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_collection_as_app(app_api_client, published_collection, permission_manage_products, channel_USD):
    if False:
        while True:
            i = 10
    collection = published_collection
    collection.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    collection.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Collection', collection.pk), 'channel': channel_USD.slug}
    response = app_api_client.post_graphql(QUERY_COLLECTION_PRIVATE_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['collection']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_DIGITAL_CONTENT_PRIVATE_META = '\n    query digitalContentMeta($id: ID!){\n        digitalContent(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_digital_content_as_anonymous_user(api_client, digital_content):
    if False:
        while True:
            i = 10
    variables = {'id': graphene.Node.to_global_id('DigitalContent', digital_content.pk)}
    response = api_client.post_graphql(QUERY_DIGITAL_CONTENT_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_digital_content_as_customer(user_api_client, digital_content):
    if False:
        return 10
    digital_content.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    digital_content.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('DigitalContent', digital_content.pk)}
    response = user_api_client.post_graphql(QUERY_DIGITAL_CONTENT_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_digital_content_as_staff(staff_api_client, digital_content, permission_manage_products):
    if False:
        i = 10
        return i + 15
    digital_content.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    digital_content.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('DigitalContent', digital_content.pk)}
    response = staff_api_client.post_graphql(QUERY_DIGITAL_CONTENT_PRIVATE_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['digitalContent']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_digital_content_as_app(app_api_client, digital_content, permission_manage_products):
    if False:
        return 10
    digital_content.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    digital_content.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('DigitalContent', digital_content.pk)}
    response = app_api_client.post_graphql(QUERY_DIGITAL_CONTENT_PRIVATE_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['digitalContent']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_TRANSACTION_ITEM_PRIVATE_META = '\nquery transactionItemMeta($id: ID!){\n  order(id: $id){\n    transactions{\n      privateMetadata{\n        key\n        value\n      }\n    }\n  }\n}\n'

def execute_query_private_metadata_for_transaction_item(client: ApiClient, order: Order, permissions: list[Permission]=None):
    if False:
        print('Hello World!')
    return execute_query(QUERY_TRANSACTION_ITEM_PRIVATE_META, client, order, 'Order', permissions)

def assert_transaction_item_contains_private_metadata(response):
    if False:
        return 10
    content = get_graphql_content(response)
    metadata = content['data']['order']['transactions'][0]['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_transaction_item_as_customer(user_api_client, order, permission_manage_orders):
    if False:
        return 10
    order.payment_transactions.create(private_metadata={PRIVATE_KEY: PRIVATE_VALUE})
    response = execute_query_public_metadata_for_transaction_item(user_api_client, order, permissions=[])
    assert_no_permission(response)

def test_query_private_meta_for_transaction_item_as_staff_with_permission(staff_api_client, order_with_lines, permission_manage_orders, permission_manage_payments):
    if False:
        for i in range(10):
            print('nop')
    order_with_lines.payment_transactions.create(private_metadata={PRIVATE_KEY: PRIVATE_VALUE})
    response = execute_query_private_metadata_for_transaction_item(staff_api_client, order_with_lines, permissions=[permission_manage_orders, permission_manage_payments])
    assert_transaction_item_contains_private_metadata(response)

def test_query_private_meta_for_transaction_item_as_staff_without_permission(staff_api_client, order):
    if False:
        for i in range(10):
            print('nop')
    order.payment_transactions.create(private_metadata={PRIVATE_KEY: PRIVATE_VALUE})
    response = execute_query_private_metadata_for_transaction_item(staff_api_client, order)
    assert_no_permission(response)

def test_query_private_meta_for_transaction_item_as_app_with_permission(app_api_client, order, permission_manage_orders, permission_manage_payments):
    if False:
        return 10
    order.payment_transactions.create(private_metadata={PRIVATE_KEY: PRIVATE_VALUE})
    response = execute_query_private_metadata_for_transaction_item(app_api_client, order, permissions=[permission_manage_payments, permission_manage_orders])
    assert_transaction_item_contains_private_metadata(response)

def test_query_private_meta_for_transaction_item_as_app_without_permission(app_api_client, order):
    if False:
        i = 10
        return i + 15
    response = execute_query_private_metadata_for_transaction_item(app_api_client, order)
    assert_no_permission(response)
QUERY_PAYMENT_PRIVATE_META = '\n    query paymentsMeta($id: ID!){\n        payment(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def execute_query_private_metadata_for_payment(client: ApiClient, payment: Payment, permissions: list[Permission]=None):
    if False:
        while True:
            i = 10
    return execute_query(QUERY_PAYMENT_PRIVATE_META, client, payment, 'Payment', permissions)

def assert_payment_contains_private_metadata(response):
    if False:
        for i in range(10):
            print('nop')
    assert_model_contains_private_metadata(response, 'payment')

def test_query_private_meta_for_payment_as_staff_with_permission(staff_api_client, payment_with_private_metadata, permission_manage_orders, permission_manage_payments):
    if False:
        while True:
            i = 10
    response = execute_query_private_metadata_for_payment(staff_api_client, payment_with_private_metadata, permissions=[permission_manage_orders, permission_manage_payments])
    assert_payment_contains_private_metadata(response)

def test_query_private_meta_for_payment_as_staff_without_permission(staff_api_client, payment_with_private_metadata):
    if False:
        print('Hello World!')
    response = execute_query_private_metadata_for_payment(staff_api_client, payment_with_private_metadata)
    assert_no_permission(response)

def test_query_private_meta_for_payment_as_app_with_permission(app_api_client, payment_with_private_metadata, permission_manage_orders, permission_manage_payments):
    if False:
        while True:
            i = 10
    response = execute_query_private_metadata_for_payment(app_api_client, payment_with_private_metadata, permissions=[permission_manage_orders, permission_manage_payments])
    assert_payment_contains_private_metadata(response)

def test_query_private_meta_for_payment_as_app_without_permission(app_api_client, payment_with_private_metadata):
    if False:
        i = 10
        return i + 15
    response = execute_query_private_metadata_for_payment(app_api_client, payment_with_private_metadata)
    assert_no_permission(response)
QUERY_PRODUCT_PRIVATE_META = '\n    query productsMeta($id: ID!, $channel: String){\n        product(id: $id, channel: $channel){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_product_as_anonymous_user(api_client, product, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': graphene.Node.to_global_id('Product', product.pk), 'channel': channel_USD.slug}
    response = api_client.post_graphql(QUERY_PRODUCT_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_product_as_customer(user_api_client, product, channel_USD):
    if False:
        return 10
    variables = {'id': graphene.Node.to_global_id('Product', product.pk), 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(QUERY_PRODUCT_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_product_as_staff(staff_api_client, product, permission_manage_products):
    if False:
        return 10
    product.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    product.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Product', product.pk)}
    response = staff_api_client.post_graphql(QUERY_PRODUCT_PRIVATE_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['product']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_product_as_app(app_api_client, product, permission_manage_products):
    if False:
        while True:
            i = 10
    product.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    product.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Product', product.pk)}
    response = app_api_client.post_graphql(QUERY_PRODUCT_PRIVATE_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['product']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_PRODUCT_TYPE_PRIVATE_META = '\n    query productTypeMeta($id: ID!){\n        productType(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_product_type_as_anonymous_user(api_client, product_type):
    if False:
        return 10
    variables = {'id': graphene.Node.to_global_id('ProductType', product_type.pk)}
    response = api_client.post_graphql(QUERY_PRODUCT_TYPE_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_product_type_as_customer(user_api_client, product_type):
    if False:
        print('Hello World!')
    variables = {'id': graphene.Node.to_global_id('ProductType', product_type.pk)}
    response = user_api_client.post_graphql(QUERY_PRODUCT_TYPE_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_product_type_as_staff(staff_api_client, product_type, permission_manage_product_types_and_attributes):
    if False:
        while True:
            i = 10
    product_type.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    product_type.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductType', product_type.pk)}
    response = staff_api_client.post_graphql(QUERY_PRODUCT_TYPE_PRIVATE_META, variables, [permission_manage_product_types_and_attributes], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['productType']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_product_type_as_app(app_api_client, product_type, permission_manage_product_types_and_attributes):
    if False:
        i = 10
        return i + 15
    product_type.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    product_type.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductType', product_type.pk)}
    response = app_api_client.post_graphql(QUERY_PRODUCT_TYPE_PRIVATE_META, variables, [permission_manage_product_types_and_attributes], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['productType']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_PRODUCT_VARIANT_PRIVATE_META = '\n    query productVariantMeta($id: ID!, $channel: String){\n        productVariant(id: $id, channel: $channel){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_product_variant_as_anonymous_user(api_client, variant, channel_USD):
    if False:
        i = 10
        return i + 15
    variant.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    variant.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductVariant', variant.pk), 'channel': channel_USD.slug}
    response = api_client.post_graphql(QUERY_PRODUCT_VARIANT_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_product_variant_as_customer(user_api_client, variant, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    variant.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    variant.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductVariant', variant.pk), 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(QUERY_PRODUCT_VARIANT_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_product_variant_as_staff(staff_api_client, variant, permission_manage_products):
    if False:
        return 10
    variant.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    variant.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductVariant', variant.pk)}
    response = staff_api_client.post_graphql(QUERY_PRODUCT_VARIANT_PRIVATE_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['productVariant']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_product_variant_as_app(app_api_client, variant, permission_manage_products):
    if False:
        print('Hello World!')
    variant.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    variant.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('ProductVariant', variant.pk)}
    response = app_api_client.post_graphql(QUERY_PRODUCT_VARIANT_PRIVATE_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['productVariant']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_APP_PRIVATE_META = '\n    query appMeta($id: ID!){\n        app(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_app_as_anonymous_user(api_client, app):
    if False:
        i = 10
        return i + 15
    variables = {'id': graphene.Node.to_global_id('App', app.pk)}
    response = api_client.post_graphql(QUERY_APP_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_app_as_customer(user_api_client, app):
    if False:
        print('Hello World!')
    variables = {'id': graphene.Node.to_global_id('App', app.pk)}
    response = user_api_client.post_graphql(QUERY_APP_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_app_as_staff(staff_api_client, app, permission_manage_apps):
    if False:
        i = 10
        return i + 15
    app.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    app.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('App', app.pk)}
    response = staff_api_client.post_graphql(QUERY_APP_PRIVATE_META, variables, [permission_manage_apps], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['app']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_app_as_app(app_api_client, app, permission_manage_apps):
    if False:
        i = 10
        return i + 15
    app.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    app.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('App', app.pk)}
    response = app_api_client.post_graphql(QUERY_APP_PRIVATE_META, variables, [permission_manage_apps], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['app']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_PAGE_TYPE_PRIVATE_META = '\n    query pageTypeMeta($id: ID!){\n        pageType(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_page_type_as_anonymous_user(api_client, page_type):
    if False:
        i = 10
        return i + 15
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = api_client.post_graphql(QUERY_PAGE_TYPE_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_page_type_as_customer(user_api_client, page_type):
    if False:
        print('Hello World!')
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = user_api_client.post_graphql(QUERY_PAGE_TYPE_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_page_type_as_staff(staff_api_client, page_type, permission_manage_page_types_and_attributes):
    if False:
        return 10
    page_type.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    page_type.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = staff_api_client.post_graphql(QUERY_PAGE_TYPE_PRIVATE_META, variables, [permission_manage_page_types_and_attributes], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['pageType']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_page_type_as_app(app_api_client, page_type, permission_manage_page_types_and_attributes):
    if False:
        while True:
            i = 10
    page_type.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    page_type.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = app_api_client.post_graphql(QUERY_PAGE_TYPE_PRIVATE_META, variables, [permission_manage_page_types_and_attributes], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['pageType']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_WAREHOUSE_PUBLIC_META = '\n    query warehouseMeta($id: ID!){\n         warehouse(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_warehouse_as_anonymous_user(api_client, warehouse):
    if False:
        while True:
            i = 10
    warehouse.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    warehouse.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Warehouse', warehouse.pk)}
    response = api_client.post_graphql(QUERY_WAREHOUSE_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_warehouse_as_customer(user_api_client, warehouse):
    if False:
        for i in range(10):
            print('nop')
    warehouse.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    warehouse.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Warehouse', warehouse.pk)}
    response = user_api_client.post_graphql(QUERY_WAREHOUSE_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_warehouse_as_staff(staff_api_client, warehouse, permission_manage_products):
    if False:
        print('Hello World!')
    warehouse.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    warehouse.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Warehouse', warehouse.pk)}
    response = staff_api_client.post_graphql(QUERY_WAREHOUSE_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['warehouse']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_warehouse_as_app(app_api_client, warehouse, permission_manage_products):
    if False:
        i = 10
        return i + 15
    warehouse.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    warehouse.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Warehouse', warehouse.pk)}
    response = app_api_client.post_graphql(QUERY_WAREHOUSE_PUBLIC_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['warehouse']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_WAREHOUSE_PRIVATE_META = '\n    query warehouseMeta($id: ID!){\n        warehouse(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_warehouse_as_anonymous_user(api_client, warehouse):
    if False:
        i = 10
        return i + 15
    variables = {'id': graphene.Node.to_global_id('Warehouse', warehouse.pk)}
    response = api_client.post_graphql(QUERY_WAREHOUSE_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_warehouse_as_customer(user_api_client, warehouse):
    if False:
        return 10
    variables = {'id': graphene.Node.to_global_id('Warehouse', warehouse.pk)}
    response = user_api_client.post_graphql(QUERY_WAREHOUSE_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_warehouse_as_staff(staff_api_client, warehouse, permission_manage_products):
    if False:
        return 10
    warehouse.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    warehouse.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Warehouse', warehouse.pk)}
    response = staff_api_client.post_graphql(QUERY_WAREHOUSE_PRIVATE_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['warehouse']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_warehouse_as_app(app_api_client, warehouse, permission_manage_products):
    if False:
        print('Hello World!')
    warehouse.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    warehouse.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Warehouse', warehouse.pk)}
    response = app_api_client.post_graphql(QUERY_WAREHOUSE_PRIVATE_META, variables, [permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['warehouse']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_VOUCHER_PUBLIC_META = '\n    query voucherMeta($id: ID!){\n         voucher(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_voucher_as_anonymous_user(api_client, voucher):
    if False:
        print('Hello World!')
    voucher.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    voucher.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = api_client.post_graphql(QUERY_VOUCHER_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_voucher_as_customer(user_api_client, voucher):
    if False:
        for i in range(10):
            print('nop')
    voucher.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    voucher.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = user_api_client.post_graphql(QUERY_VOUCHER_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_voucher_as_staff(staff_api_client, voucher, permission_manage_discounts):
    if False:
        while True:
            i = 10
    voucher.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    voucher.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = staff_api_client.post_graphql(QUERY_VOUCHER_PUBLIC_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['voucher']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_voucher_as_app(app_api_client, voucher, permission_manage_discounts):
    if False:
        i = 10
        return i + 15
    voucher.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    voucher.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = app_api_client.post_graphql(QUERY_VOUCHER_PUBLIC_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['voucher']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_VOUCHER_PRIVATE_META = '\n    query voucherMeta($id: ID!){\n        voucher(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_voucher_as_anonymous_user(api_client, voucher):
    if False:
        print('Hello World!')
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = api_client.post_graphql(QUERY_VOUCHER_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_voucher_as_customer(user_api_client, voucher):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = user_api_client.post_graphql(QUERY_VOUCHER_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_voucher_as_staff(staff_api_client, voucher, permission_manage_discounts):
    if False:
        i = 10
        return i + 15
    voucher.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    voucher.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = staff_api_client.post_graphql(QUERY_VOUCHER_PRIVATE_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['voucher']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_voucher_as_app(app_api_client, voucher, permission_manage_discounts):
    if False:
        print('Hello World!')
    voucher.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    voucher.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = app_api_client.post_graphql(QUERY_VOUCHER_PRIVATE_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['voucher']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_SALE_PUBLIC_META = '\n    query saleMeta($id: ID!){\n         sale(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_sale_as_anonymous_user(api_client, promotion_converted_from_sale):
    if False:
        for i in range(10):
            print('nop')
    sale = promotion_converted_from_sale
    sale.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    sale.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Sale', sale.old_sale_id)}
    response = api_client.post_graphql(QUERY_SALE_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_sale_as_customer(user_api_client, promotion_converted_from_sale):
    if False:
        i = 10
        return i + 15
    sale = promotion_converted_from_sale
    sale.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    sale.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Sale', sale.old_sale_id)}
    response = user_api_client.post_graphql(QUERY_SALE_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_sale_as_staff(staff_api_client, promotion_converted_from_sale, permission_manage_discounts):
    if False:
        for i in range(10):
            print('nop')
    sale = promotion_converted_from_sale
    sale.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    sale.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Sale', sale.old_sale_id)}
    response = staff_api_client.post_graphql(QUERY_SALE_PUBLIC_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['sale']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_sale_as_app(app_api_client, promotion_converted_from_sale, permission_manage_discounts):
    if False:
        for i in range(10):
            print('nop')
    sale = promotion_converted_from_sale
    sale.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    sale.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Sale', sale.old_sale_id)}
    response = app_api_client.post_graphql(QUERY_SALE_PUBLIC_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['sale']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_SALE_PRIVATE_META = '\n    query saleMeta($id: ID!){\n        sale(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_sale_as_anonymous_user(api_client, promotion_converted_from_sale):
    if False:
        print('Hello World!')
    sale = promotion_converted_from_sale
    variables = {'id': graphene.Node.to_global_id('Sale', sale.old_sale_id)}
    response = api_client.post_graphql(QUERY_SALE_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_sale_as_customer(user_api_client, promotion_converted_from_sale):
    if False:
        i = 10
        return i + 15
    sale = promotion_converted_from_sale
    variables = {'id': graphene.Node.to_global_id('Sale', sale.old_sale_id)}
    response = user_api_client.post_graphql(QUERY_SALE_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_sale_as_staff(staff_api_client, promotion_converted_from_sale, permission_manage_discounts):
    if False:
        print('Hello World!')
    sale = promotion_converted_from_sale
    sale.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    sale.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Sale', sale.old_sale_id)}
    response = staff_api_client.post_graphql(QUERY_SALE_PRIVATE_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['sale']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_sale_as_app(app_api_client, promotion_converted_from_sale, permission_manage_discounts):
    if False:
        return 10
    sale = promotion_converted_from_sale
    sale.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    sale.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Sale', sale.old_sale_id)}
    response = app_api_client.post_graphql(QUERY_SALE_PRIVATE_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['sale']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_GIFT_CARD_PRIVATE_META = '\n    query giftCardMeta($id: ID!){\n        giftCard(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_gift_card_as_anonymous_user(api_client, gift_card):
    if False:
        while True:
            i = 10
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = api_client.post_graphql(QUERY_GIFT_CARD_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_gift_card_as_customer(user_api_client, gift_card):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = user_api_client.post_graphql(QUERY_GIFT_CARD_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_gift_card_as_staff(staff_api_client, gift_card, permission_manage_gift_card):
    if False:
        i = 10
        return i + 15
    gift_card.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    gift_card.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = staff_api_client.post_graphql(QUERY_GIFT_CARD_PRIVATE_META, variables, [permission_manage_gift_card], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['giftCard']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_gift_card_as_app(app_api_client, gift_card, permission_manage_gift_card):
    if False:
        i = 10
        return i + 15
    gift_card.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    gift_card.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = app_api_client.post_graphql(QUERY_GIFT_CARD_PRIVATE_META, variables, [permission_manage_gift_card], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['giftCard']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_PRODUCT_MEDIA_METADATA = '\n    query productMediaById(\n        $mediaId: ID!,\n        $productId: ID!,\n        $channel: String,\n    ) {\n        product(id: $productId, channel: $channel) {\n            mediaById(id: $mediaId) {\n                metadata {\n                    key\n                    value\n                }\n                privateMetadata {\n                    key\n                    value\n                }\n            }\n        }\n    }\n'

def test_query_metadata_for_product_media_as_staff(staff_api_client, product_with_image, channel_USD, permission_manage_products):
    if False:
        print('Hello World!')
    query = QUERY_PRODUCT_MEDIA_METADATA
    media = product_with_image.media.first()
    metadata = {'label': 'image-name'}
    private_metadata = {'private-label': 'private-name'}
    media.store_value_in_metadata(metadata)
    media.store_value_in_private_metadata(private_metadata)
    media.save(update_fields=['metadata', 'private_metadata'])
    variables = {'productId': graphene.Node.to_global_id('Product', product_with_image.pk), 'mediaId': graphene.Node.to_global_id('ProductMedia', media.pk), 'channel': channel_USD.slug}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    data = content['data']['product']['mediaById']
    assert data['metadata'][0]['key'] in metadata.keys()
    assert data['metadata'][0]['value'] in metadata.values()
    assert data['privateMetadata'][0]['key'] in private_metadata.keys()
    assert data['privateMetadata'][0]['value'] in private_metadata.values()

def test_query_metadata_for_product_media_as_app(app_api_client, product_with_image, channel_USD, permission_manage_products):
    if False:
        i = 10
        return i + 15
    query = QUERY_PRODUCT_MEDIA_METADATA
    media = product_with_image.media.first()
    metadata = {'label': 'image-name'}
    private_metadata = {'private-label': 'private-name'}
    media.store_value_in_metadata(metadata)
    media.store_value_in_private_metadata(private_metadata)
    media.save(update_fields=['metadata', 'private_metadata'])
    variables = {'productId': graphene.Node.to_global_id('Product', product_with_image.pk), 'mediaId': graphene.Node.to_global_id('ProductMedia', media.pk), 'channel': channel_USD.slug}
    response = app_api_client.post_graphql(query, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    data = content['data']['product']['mediaById']
    assert data['metadata'][0]['key'] in metadata.keys()
    assert data['metadata'][0]['value'] in metadata.values()
    assert data['privateMetadata'][0]['key'] in private_metadata.keys()
    assert data['privateMetadata'][0]['value'] in private_metadata.values()

def test_query_metadata_for_product_media_as_anonymous_user(api_client, product_with_image, channel_USD):
    if False:
        print('Hello World!')
    query = QUERY_PRODUCT_MEDIA_METADATA
    media = product_with_image.media.first()
    variables = {'productId': graphene.Node.to_global_id('Product', product_with_image.pk), 'mediaId': graphene.Node.to_global_id('ProductMedia', media.pk), 'channel': channel_USD.slug}
    response = api_client.post_graphql(query, variables)
    assert_no_permission(response)

def test_query_metadata_for_product_media_as_customer_user(user_api_client, product_with_image, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    query = QUERY_PRODUCT_MEDIA_METADATA
    media = product_with_image.media.first()
    variables = {'productId': graphene.Node.to_global_id('Product', product_with_image.pk), 'mediaId': graphene.Node.to_global_id('ProductMedia', media.pk), 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(query, variables)
    assert_no_permission(response)

def test_query_metadata_for_product_media_as_staff_missing_permissions(staff_api_client, product_with_image, channel_USD):
    if False:
        return 10
    query = QUERY_PRODUCT_MEDIA_METADATA
    media = product_with_image.media.first()
    variables = {'productId': graphene.Node.to_global_id('Product', product_with_image.pk), 'mediaId': graphene.Node.to_global_id('ProductMedia', media.pk), 'channel': channel_USD.slug}
    response = staff_api_client.post_graphql(query, variables)
    assert_no_permission(response)