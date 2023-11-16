import graphene
from ....tests.utils import assert_no_permission, get_graphql_content, get_graphql_content_from_response
QUERY_SALE_BY_ID = '\n    query Sale($id: ID!, $channel: String) {\n        sale(id: $id, channel: $channel) {\n            id\n            name\n            type\n            discountValue\n            currency\n            collections(first: 1) {\n                edges {\n                    node {\n                        name\n                    }\n                }\n            }\n            categories(first: 1) {\n                edges {\n                    node {\n                        name\n                    }\n                }\n            }\n            products(first: 1) {\n                edges {\n                    node {\n                        name\n                    }\n                }\n            }\n            variants(first: 1) {\n                edges {\n                    node {\n                        name\n                    }\n                }\n            }\n            channelListings {\n                id\n                discountValue\n                channel {\n                    slug\n                }\n                currency\n            }\n        }\n    }\n'

def test_staff_query_sale(staff_api_client, promotion_converted_from_sale, permission_manage_discounts, product, variant, collection, category, channel_USD):
    if False:
        print('Hello World!')
    promotion = promotion_converted_from_sale
    rule = promotion.rules.first()
    channel = rule.channels.first()
    variables = {'id': graphene.Node.to_global_id('Sale', promotion.old_sale_id), 'channel': channel_USD.slug}
    response = staff_api_client.post_graphql(QUERY_SALE_BY_ID, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    sale_data = content['data']['sale']
    assert sale_data['id'] == graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    assert sale_data['name'] == promotion.name
    assert sale_data['type'] == rule.reward_value_type.upper()
    assert sale_data['discountValue'] == rule.reward_value
    assert sale_data['currency'] == channel.currency_code
    assert sale_data['products']['edges'][0]['node']['name'] == product.name
    assert sale_data['variants']['edges'][0]['node']['name'] == variant.name
    assert sale_data['collections']['edges'][0]['node']['name'] == collection.name
    assert sale_data['categories']['edges'][0]['node']['name'] == category.name
    channel_listing = sale_data['channelListings'][0]
    assert channel_listing['discountValue'] == rule.reward_value
    assert channel_listing['channel']['slug'] == channel.slug
    assert channel_listing['currency'] == channel.currency_code
    assert channel_listing['id'] == graphene.Node.to_global_id('SaleChannelListing', rule.old_channel_listing_id)

def test_query_sale_by_app(app_api_client, promotion_converted_from_sale, permission_manage_discounts):
    if False:
        i = 10
        return i + 15
    promotion = promotion_converted_from_sale
    variables = {'id': graphene.Node.to_global_id('Sale', promotion.old_sale_id)}
    response = app_api_client.post_graphql(QUERY_SALE_BY_ID, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    assert content['data']['sale']['name'] == promotion.name
    assert content['data']['sale']['type'] == promotion.rules.first().reward_value_type.upper()

def test_query_sale_by_customer(api_client, promotion_converted_from_sale):
    if False:
        for i in range(10):
            print('nop')
    promotion = promotion_converted_from_sale
    variables = {'id': graphene.Node.to_global_id('Sale', promotion.old_sale_id)}
    response = api_client.post_graphql(QUERY_SALE_BY_ID, variables)
    assert_no_permission(response)

def test_staff_query_sale_by_invalid_id(staff_api_client, promotion_converted_from_sale, permission_manage_discounts):
    if False:
        while True:
            i = 10
    id = 'bh/'
    variables = {'id': id}
    response = staff_api_client.post_graphql(QUERY_SALE_BY_ID, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content_from_response(response)
    assert len(content['errors']) == 1
    assert content['errors'][0]['message'] == f'Invalid ID: {id}. Expected: Sale.'
    assert content['data']['sale'] is None

def test_staff_query_sale_with_invalid_object_type(staff_api_client, promotion_converted_from_sale, permission_manage_discounts):
    if False:
        while True:
            i = 10
    promotion = promotion_converted_from_sale
    variables = {'id': graphene.Node.to_global_id('Order', promotion.old_sale_id)}
    response = staff_api_client.post_graphql(QUERY_SALE_BY_ID, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    assert content['data']['sale'] is None

def test_staff_query_sale_no_channel_provided(staff_api_client, promotion_converted_from_sale, permission_manage_discounts):
    if False:
        i = 10
        return i + 15
    promotion = promotion_converted_from_sale
    rule = promotion.rules.first()
    variables = {'id': graphene.Node.to_global_id('Sale', promotion.old_sale_id)}
    response = staff_api_client.post_graphql(QUERY_SALE_BY_ID, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    sale_data = content['data']['sale']
    assert sale_data['type'] == rule.reward_value_type.upper()
    assert not sale_data['discountValue']
    assert not sale_data['currency']