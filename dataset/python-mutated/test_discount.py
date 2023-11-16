import graphene
from ....tests.utils import assert_no_permission, get_graphql_content
from .utils import PRIVATE_KEY, PRIVATE_VALUE, PUBLIC_KEY, PUBLIC_VALUE
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
        print('Hello World!')
    sale = promotion_converted_from_sale
    sale.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    sale.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Sale', sale.old_sale_id)}
    response = user_api_client.post_graphql(QUERY_SALE_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_sale_as_staff(staff_api_client, promotion_converted_from_sale, permission_manage_discounts):
    if False:
        return 10
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
        return 10
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
        for i in range(10):
            print('nop')
    sale = promotion_converted_from_sale
    variables = {'id': graphene.Node.to_global_id('Sale', sale.old_sale_id)}
    response = api_client.post_graphql(QUERY_SALE_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_sale_as_customer(user_api_client, promotion_converted_from_sale):
    if False:
        print('Hello World!')
    sale = promotion_converted_from_sale
    variables = {'id': graphene.Node.to_global_id('Sale', sale.old_sale_id)}
    response = user_api_client.post_graphql(QUERY_SALE_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_sale_as_staff(staff_api_client, promotion_converted_from_sale, permission_manage_discounts):
    if False:
        i = 10
        return i + 15
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
        while True:
            i = 10
    sale = promotion_converted_from_sale
    sale.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    sale.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Sale', sale.old_sale_id)}
    response = app_api_client.post_graphql(QUERY_SALE_PRIVATE_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['sale']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_VOUCHER_PUBLIC_META = '\n    query voucherMeta($id: ID!){\n         voucher(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_voucher_as_anonymous_user(api_client, voucher):
    if False:
        return 10
    voucher.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    voucher.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = api_client.post_graphql(QUERY_VOUCHER_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_voucher_as_customer(user_api_client, voucher):
    if False:
        while True:
            i = 10
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
        while True:
            i = 10
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
        for i in range(10):
            print('nop')
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = api_client.post_graphql(QUERY_VOUCHER_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_voucher_as_customer(user_api_client, voucher):
    if False:
        return 10
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
        while True:
            i = 10
    voucher.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    voucher.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = app_api_client.post_graphql(QUERY_VOUCHER_PRIVATE_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['voucher']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_PROMOTION_PUBLIC_META = '\n    query promotionMeta($id: ID!){\n         promotion(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_promotion_as_anonymous_user(api_client, promotion):
    if False:
        for i in range(10):
            print('nop')
    promotion.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    promotion.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Promotion', promotion.pk)}
    response = api_client.post_graphql(QUERY_PROMOTION_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_promotion_as_customer(user_api_client, promotion):
    if False:
        for i in range(10):
            print('nop')
    promotion.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    promotion.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Promotion', promotion.pk)}
    response = user_api_client.post_graphql(QUERY_PROMOTION_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_promotion_as_staff(staff_api_client, promotion, permission_manage_discounts):
    if False:
        i = 10
        return i + 15
    promotion.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    promotion.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Promotion', promotion.pk)}
    response = staff_api_client.post_graphql(QUERY_PROMOTION_PUBLIC_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['promotion']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_promotion_as_app(app_api_client, promotion, permission_manage_discounts):
    if False:
        i = 10
        return i + 15
    promotion.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    promotion.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('Promotion', promotion.pk)}
    response = app_api_client.post_graphql(QUERY_PROMOTION_PUBLIC_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['promotion']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE
QUERY_PROMOTION_PRIVATE_META = '\n    query promotionMeta($id: ID!){\n        promotion(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_promotion_as_anonymous_user(api_client, promotion):
    if False:
        while True:
            i = 10
    variables = {'id': graphene.Node.to_global_id('Promotion', promotion.pk)}
    response = api_client.post_graphql(QUERY_PROMOTION_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_promotion_as_customer(user_api_client, promotion):
    if False:
        return 10
    variables = {'id': graphene.Node.to_global_id('Promotion', promotion.pk)}
    response = user_api_client.post_graphql(QUERY_PROMOTION_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_promotion_as_staff(staff_api_client, promotion, permission_manage_discounts):
    if False:
        while True:
            i = 10
    promotion.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    promotion.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Promotion', promotion.pk)}
    response = staff_api_client.post_graphql(QUERY_PROMOTION_PRIVATE_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['promotion']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_promotion_as_app(app_api_client, promotion, permission_manage_discounts):
    if False:
        while True:
            i = 10
    promotion.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    promotion.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('Promotion', promotion.pk)}
    response = app_api_client.post_graphql(QUERY_PROMOTION_PRIVATE_META, variables, [permission_manage_discounts], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['promotion']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE