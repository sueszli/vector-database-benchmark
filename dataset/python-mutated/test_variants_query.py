import graphene
from .....product.models import ProductVariant
from ....tests.utils import assert_no_permission, get_graphql_content

def test_product_variants_by_ids(staff_api_client, variant, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    query = '\n        query getProductVariants($ids: [ID!], $channel: String) {\n            productVariants(ids: $ids, first: 1, channel: $channel) {\n                edges {\n                    node {\n                        id\n                        name\n                        sku\n                        channelListings {\n                            channel {\n                                id\n                                isActive\n                                name\n                                currencyCode\n                            }\n                            price {\n                                amount\n                                currency\n                            }\n                        }\n                    }\n                }\n            }\n        }\n    '
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.id)
    variables = {'ids': [variant_id], 'channel': channel_USD.slug}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['productVariants']
    assert data['edges'][0]['node']['id'] == variant_id
    assert len(data['edges']) == 1

def test_product_variants_without_price_by_ids_as_staff_without_permission(staff_api_client, variant, channel_USD):
    if False:
        while True:
            i = 10
    query = '\n        query getProductVariants($ids: [ID!], $channel: String) {\n            productVariants(ids: $ids, first: 1, channel: $channel) {\n                edges {\n                    node {\n                        id\n                        name\n                        sku\n                        channelListings {\n                            channel {\n                                id\n                                isActive\n                                name\n                                currencyCode\n                            }\n                            price {\n                                amount\n                                currency\n                            }\n                        }\n                    }\n                }\n            }\n        }\n    '
    variant.channel_listings.all().delete()
    variant.channel_listings.create(channel=channel_USD)
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.id)
    variables = {'ids': [variant_id], 'channel': channel_USD.slug}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['productVariants']
    assert len(data['edges']) == 0

def test_product_variants_without_price_by_ids_as_staff_with_permission(staff_api_client, variant, channel_USD, permission_manage_products):
    if False:
        i = 10
        return i + 15
    query = '\n        query getProductVariants($ids: [ID!], $channel: String) {\n            productVariants(ids: $ids, first: 1, channel: $channel) {\n                edges {\n                    node {\n                        id\n                        name\n                        sku\n                        channelListings {\n                            channel {\n                                id\n                                isActive\n                                name\n                                currencyCode\n                            }\n                            price {\n                                amount\n                                currency\n                            }\n                        }\n                    }\n                }\n            }\n        }\n    '
    variant.channel_listings.all().delete()
    variant.channel_listings.create(channel=channel_USD)
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.id)
    variables = {'ids': [variant_id], 'channel': channel_USD.slug}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    data = content['data']['productVariants']
    assert data['edges'][0]['node']['id'] == variant_id
    assert len(data['edges']) == 1

def test_product_variants_without_price_by_ids_as_user(user_api_client, variant, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    query = '\n        query getProductVariants($ids: [ID!], $channel: String) {\n            productVariants(ids: $ids, first: 1, channel: $channel) {\n                edges {\n                    node {\n                        id\n                        name\n                        sku\n                    }\n                }\n            }\n        }\n    '
    variant.channel_listings.all().delete()
    variant.channel_listings.create(channel=channel_USD)
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.id)
    variables = {'ids': [variant_id], 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['productVariants']
    assert len(data['edges']) == 0

def test_product_variants_without_price_by_ids_as_app_without_permission(app_api_client, variant, channel_USD):
    if False:
        i = 10
        return i + 15
    query = '\n        query getProductVariants($ids: [ID!], $channel: String) {\n            productVariants(ids: $ids, first: 1, channel: $channel) {\n                edges {\n                    node {\n                        id\n                        name\n                        sku\n                        channelListings {\n                            channel {\n                                id\n                                isActive\n                                name\n                                currencyCode\n                            }\n                            price {\n                                amount\n                                currency\n                            }\n                        }\n                    }\n                }\n            }\n        }\n    '
    variant.channel_listings.all().delete()
    variant.channel_listings.create(channel=channel_USD)
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.id)
    variables = {'ids': [variant_id], 'channel': channel_USD.slug}
    response = app_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    assert len(content['data']['productVariants']['edges']) == 0

def test_product_variants_without_price_by_ids_as_app_with_permission(app_api_client, variant, channel_USD, permission_manage_products):
    if False:
        i = 10
        return i + 15
    query = '\n        query getProductVariants($ids: [ID!], $channel: String) {\n            productVariants(ids: $ids, first: 1, channel: $channel) {\n                edges {\n                    node {\n                        id\n                        name\n                        sku\n                        channelListings {\n                            channel {\n                                id\n                                isActive\n                                name\n                                currencyCode\n                            }\n                            price {\n                                amount\n                                currency\n                            }\n                        }\n                    }\n                }\n            }\n        }\n    '
    variant.channel_listings.all().delete()
    variant.channel_listings.create(channel=channel_USD)
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.id)
    variables = {'ids': [variant_id], 'channel': channel_USD.slug}
    response = app_api_client.post_graphql(query, variables, permissions=[permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    data = content['data']['productVariants']
    assert data['edges'][0]['node']['id'] == variant_id
    assert len(data['edges']) == 1

def test_product_variants_by_customer(user_api_client, variant, channel_USD):
    if False:
        print('Hello World!')
    query = '\n        query getProductVariants($ids: [ID!], $channel: String) {\n            productVariants(ids: $ids, first: 1, channel: $channel) {\n                edges {\n                    node {\n                        id\n                        name\n                        sku\n                        channelListings {\n                            channel {\n                                id\n                                isActive\n                                name\n                                currencyCode\n                            }\n                            price {\n                                amount\n                                currency\n                            }\n                        }\n                    }\n                }\n            }\n        }\n    '
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.id)
    variables = {'ids': [variant_id], 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(query, variables)
    assert_no_permission(response)

def test_product_variants_no_ids_list(user_api_client, variant, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    query = '\n        query getProductVariants($channel: String) {\n            productVariants(first: 10, channel: $channel) {\n                edges {\n                    node {\n                        id\n                    }\n                }\n            }\n        }\n    '
    variables = {'channel': channel_USD.slug}
    response = user_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['productVariants']
    assert len(data['edges']) == ProductVariant.objects.count()