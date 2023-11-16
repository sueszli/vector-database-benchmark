import graphene
from django.contrib.sites.models import Site
from measurement.measures import Weight
from .....core.units import WeightUnits
from ....core.enums import WeightUnitsEnum
from ....tests.utils import assert_no_permission, get_graphql_content
QUERY_VARIANT = 'query ProductVariantDetails(\n        $id: ID!, $address: AddressInput, $countryCode: CountryCode, $channel: String\n    ) {\n        productVariant(id: $id, channel: $channel) {\n            id\n            deprecatedStocksByCountry: stocks(countryCode: $countryCode) {\n                id\n            }\n            stocksByAddress: stocks(address: $address) {\n                id\n            }\n            attributes {\n                attribute {\n                    id\n                    name\n                    slug\n                    choices(first: 10) {\n                        edges {\n                            node {\n                                id\n                                name\n                                slug\n                            }\n                        }\n                    }\n                }\n                values {\n                    id\n                    name\n                    slug\n                }\n            }\n            media {\n                id\n            }\n            name\n            channelListings {\n                channel {\n                    slug\n                }\n                price {\n                    currency\n                    amount\n                }\n                costPrice {\n                    currency\n                    amount\n                }\n            }\n            product {\n                id\n            }\n            weight {\n                unit\n                value\n            }\n            created\n        }\n    }\n'

def test_fetch_variant(staff_api_client, product, permission_manage_products, site_settings, settings, channel_USD):
    if False:
        i = 10
        return i + 15
    query = QUERY_VARIANT
    variant = product.variants.first()
    variant.weight = Weight(kg=10)
    variant.save(update_fields=['weight'])
    site_settings.default_weight_unit = WeightUnits.G
    site_settings.save(update_fields=['default_weight_unit'])
    Site.objects.clear_cache()
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    variables = {'id': variant_id, 'countryCode': 'EU', 'channel': channel_USD.slug}
    staff_api_client.user.user_permissions.add(permission_manage_products)
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['productVariant']
    assert data['name'] == variant.name
    assert data['created'] == variant.created_at.isoformat()
    stocks_count = variant.stocks.count()
    assert len(data['deprecatedStocksByCountry']) == stocks_count
    assert len(data['stocksByAddress']) == stocks_count
    assert data['weight']['value'] == 10000
    assert data['weight']['unit'] == WeightUnitsEnum.G.name
    channel_listing_data = data['channelListings'][0]
    channel_listing = variant.channel_listings.get()
    assert channel_listing_data['channel']['slug'] == channel_listing.channel.slug
    assert channel_listing_data['price']['currency'] == channel_listing.currency
    assert channel_listing_data['price']['amount'] == channel_listing.price_amount
    assert channel_listing_data['costPrice']['currency'] == channel_listing.currency
    assert channel_listing_data['costPrice']['amount'] == channel_listing.cost_price_amount

def test_fetch_variant_no_stocks(staff_api_client, product, permission_manage_products, site_settings, channel_USD):
    if False:
        return 10
    query = QUERY_VARIANT
    variant = product.variants.first()
    variant.weight = Weight(kg=10)
    variant.save(update_fields=['weight'])
    site_settings.default_weight_unit = WeightUnits.G
    site_settings.save(update_fields=['default_weight_unit'])
    Site.objects.clear_cache()
    warehouse = variant.stocks.first().warehouse
    warehouse.channels.clear()
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    variables = {'id': variant_id, 'countryCode': 'EU', 'channel': channel_USD.slug}
    staff_api_client.user.user_permissions.add(permission_manage_products)
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['productVariant']
    assert data['name'] == variant.name
    assert data['created'] == variant.created_at.isoformat()
    assert not data['deprecatedStocksByCountry']
    assert not data['stocksByAddress']
    assert data['weight']['value'] == 10000
    assert data['weight']['unit'] == WeightUnitsEnum.G.name
    channel_listing_data = data['channelListings'][0]
    channel_listing = variant.channel_listings.get()
    assert channel_listing_data['channel']['slug'] == channel_listing.channel.slug
    assert channel_listing_data['price']['currency'] == channel_listing.currency
    assert channel_listing_data['price']['amount'] == channel_listing.price_amount
    assert channel_listing_data['costPrice']['currency'] == channel_listing.currency
    assert channel_listing_data['costPrice']['amount'] == channel_listing.cost_price_amount
QUERY_PRODUCT_VARIANT_CHANNEL_LISTING = '\n    query ProductVariantDetails($id: ID!, $channel: String) {\n        productVariant(id: $id, channel: $channel) {\n            id\n            channelListings {\n                channel {\n                    slug\n                }\n                price {\n                    currency\n                    amount\n                }\n                costPrice {\n                    currency\n                    amount\n                }\n                preorderThreshold {\n                    quantity\n                    soldUnits\n                }\n            }\n        }\n    }\n'

def test_get_product_variant_channel_listing_as_staff_user(staff_api_client, product_available_in_many_channels, channel_USD):
    if False:
        while True:
            i = 10
    variant = product_available_in_many_channels.variants.get()
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    variables = {'id': variant_id, 'channel': channel_USD.slug}
    response = staff_api_client.post_graphql(QUERY_PRODUCT_VARIANT_CHANNEL_LISTING, variables)
    content = get_graphql_content(response)
    data = content['data']['productVariant']
    channel_listings = variant.channel_listings.all()
    for channel_listing in channel_listings:
        assert {'channel': {'slug': channel_listing.channel.slug}, 'price': {'currency': channel_listing.currency, 'amount': channel_listing.price_amount}, 'costPrice': {'currency': channel_listing.currency, 'amount': channel_listing.cost_price_amount}, 'preorderThreshold': {'quantity': channel_listing.preorder_quantity_threshold, 'soldUnits': 0}} in data['channelListings']
    assert len(data['channelListings']) == variant.channel_listings.count()

def test_get_product_variant_channel_listing_as_app(app_api_client, product_available_in_many_channels, channel_USD):
    if False:
        return 10
    variant = product_available_in_many_channels.variants.get()
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    variables = {'id': variant_id, 'channel': channel_USD.slug}
    response = app_api_client.post_graphql(QUERY_PRODUCT_VARIANT_CHANNEL_LISTING, variables)
    content = get_graphql_content(response)
    data = content['data']['productVariant']
    channel_listings = variant.channel_listings.all()
    for channel_listing in channel_listings:
        assert {'channel': {'slug': channel_listing.channel.slug}, 'price': {'currency': channel_listing.currency, 'amount': channel_listing.price_amount}, 'costPrice': {'currency': channel_listing.currency, 'amount': channel_listing.cost_price_amount}, 'preorderThreshold': {'quantity': channel_listing.preorder_quantity_threshold, 'soldUnits': 0}} in data['channelListings']
    assert len(data['channelListings']) == variant.channel_listings.count()

def test_get_product_variant_channel_listing_as_customer(user_api_client, product_available_in_many_channels, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    variant = product_available_in_many_channels.variants.get()
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    variables = {'id': variant_id, 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(QUERY_PRODUCT_VARIANT_CHANNEL_LISTING, variables)
    assert_no_permission(response)

def test_get_product_variant_channel_listing_as_anonymous(api_client, product_available_in_many_channels, channel_USD):
    if False:
        while True:
            i = 10
    variant = product_available_in_many_channels.variants.get()
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    variables = {'id': variant_id, 'channel': channel_USD.slug}
    response = api_client.post_graphql(QUERY_PRODUCT_VARIANT_CHANNEL_LISTING, variables)
    assert_no_permission(response)
QUERY_PRODUCT_VARIANT_STOCKS = '\n  fragment Stock on Stock {\n    id\n    quantity\n    warehouse {\n      slug\n    }\n  }\n  query ProductVariantDetails(\n    $id: ID!\n    $channel: String\n    $address: AddressInput\n  ) {\n    productVariant(id: $id, channel: $channel) {\n      id\n      stocksNoAddress: stocks {\n        ...Stock\n      }\n      stocksWithAddress: stocks(address: $address) {\n        ...Stock\n      }\n    }\n  }\n'

def test_get_product_variant_stocks(staff_api_client, variant_with_many_stocks_different_shipping_zones, channel_USD, permission_manage_products):
    if False:
        i = 10
        return i + 15
    variant = variant_with_many_stocks_different_shipping_zones
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    variables = {'id': variant_id, 'channel': channel_USD.slug, 'address': {'country': 'PL'}}
    response = staff_api_client.post_graphql(QUERY_PRODUCT_VARIANT_STOCKS, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    all_stocks = variant.stocks.all()
    pl_stocks = variant.stocks.filter(warehouse__shipping_zones__countries__contains='PL')
    data = content['data']['productVariant']
    assert len(data['stocksNoAddress']) == all_stocks.count()
    no_address_stocks_ids = [stock['id'] for stock in data['stocksNoAddress']]
    assert all([graphene.Node.to_global_id('Stock', stock.pk) in no_address_stocks_ids for stock in all_stocks])
    assert len(data['stocksWithAddress']) == pl_stocks.count()
    with_address_stocks_ids = [stock['id'] for stock in data['stocksWithAddress']]
    assert all([graphene.Node.to_global_id('Stock', stock.pk) in with_address_stocks_ids for stock in pl_stocks])

def test_get_product_variant_stocks_no_channel_shipping_zones(staff_api_client, variant_with_many_stocks_different_shipping_zones, channel_USD, permission_manage_products):
    if False:
        i = 10
        return i + 15
    channel_USD.shipping_zones.clear()
    variant = variant_with_many_stocks_different_shipping_zones
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    variables = {'id': variant_id, 'channel': channel_USD.slug, 'address': {'country': 'PL'}}
    response = staff_api_client.post_graphql(QUERY_PRODUCT_VARIANT_STOCKS, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    stocks_count = variant.stocks.count()
    data = content['data']['productVariant']
    assert data['stocksNoAddress'] == []
    assert data['stocksWithAddress'] == []
    assert stocks_count > 0
QUERY_PRODUCT_VARIANT_PREORDER = '\n    query ProductVariantDetails($id: ID!, $channel: String) {\n        productVariant(id: $id, channel: $channel) {\n            preorder {\n                globalThreshold\n                globalSoldUnits\n                endDate\n            }\n        }\n    }\n'

def test_get_product_variant_preorder_as_staff(staff_api_client, preorder_variant_global_and_channel_threshold, preorder_allocation, channel_USD, permission_manage_products):
    if False:
        print('Hello World!')
    variant = preorder_variant_global_and_channel_threshold
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    variables = {'id': variant_id, 'channel': channel_USD.slug}
    response = staff_api_client.post_graphql(QUERY_PRODUCT_VARIANT_PREORDER, variables, permissions=[permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    data = content['data']['productVariant']['preorder']
    assert data['globalThreshold'] == variant.preorder_global_threshold
    assert data['globalSoldUnits'] == preorder_allocation.quantity
    assert data['endDate'] == variant.preorder_end_date

def test_get_product_variant_preorder_as_customer_not_allowed_fields(user_api_client, preorder_variant_global_threshold, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    variant = preorder_variant_global_threshold
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    variables = {'id': variant_id, 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(QUERY_PRODUCT_VARIANT_PREORDER, variables)
    assert_no_permission(response)

def test_get_product_variant_preorder_as_customer_allowed_fields(user_api_client, preorder_variant_global_threshold, channel_USD):
    if False:
        return 10
    variant = preorder_variant_global_threshold
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    query = '\n        query ProductVariantDetails($id: ID!, $channel: String) {\n            productVariant(id: $id, channel: $channel) {\n                preorder {\n                    endDate\n                }\n            }\n        }\n    '
    variables = {'id': variant_id, 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['productVariant']['preorder']
    assert data['endDate'] == variant.preorder_end_date

def _fetch_variant(client, variant, channel_slug=None, permissions=None):
    if False:
        return 10
    query = '\n    query ProductVariantDetails($variantId: ID!, $channel: String) {\n        productVariant(id: $variantId, channel: $channel) {\n            id\n            product {\n                id\n            }\n        }\n    }\n    '
    variables = {'variantId': graphene.Node.to_global_id('ProductVariant', variant.id)}
    if channel_slug:
        variables['channel'] = channel_slug
    response = client.post_graphql(query, variables, permissions=permissions, check_no_permissions=False)
    content = get_graphql_content(response)
    return content['data']['productVariant']

def test_fetch_unpublished_variant_staff_user(staff_api_client, unavailable_product_with_variant, permission_manage_products):
    if False:
        return 10
    variant = unavailable_product_with_variant.variants.first()
    data = _fetch_variant(staff_api_client, variant, permissions=[permission_manage_products])
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    product_id = graphene.Node.to_global_id('Product', unavailable_product_with_variant.pk)
    assert data['id'] == variant_id
    assert data['product']['id'] == product_id

def test_fetch_unpublished_variant_customer(user_api_client, unavailable_product_with_variant, channel_USD):
    if False:
        return 10
    variant = unavailable_product_with_variant.variants.first()
    data = _fetch_variant(user_api_client, variant, channel_slug=channel_USD.slug)
    assert data is None

def test_fetch_unpublished_variant_anonymous_user(api_client, unavailable_product_with_variant, channel_USD):
    if False:
        while True:
            i = 10
    variant = unavailable_product_with_variant.variants.first()
    data = _fetch_variant(api_client, variant, channel_slug=channel_USD.slug)
    assert data is None

def test_fetch_variant_without_sku_staff_user(staff_api_client, product, variant, permission_manage_products):
    if False:
        print('Hello World!')
    variant.sku = None
    variant.save()
    data = _fetch_variant(staff_api_client, variant, permissions=[permission_manage_products])
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    product_id = graphene.Node.to_global_id('Product', product.pk)
    assert data['id'] == variant_id
    assert data['product']['id'] == product_id

def test_fetch_variant_without_sku_customer(user_api_client, product, variant, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    variant.sku = None
    variant.save()
    data = _fetch_variant(user_api_client, variant, channel_slug=channel_USD.slug)
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    product_id = graphene.Node.to_global_id('Product', product.pk)
    assert data['id'] == variant_id
    assert data['product']['id'] == product_id

def test_fetch_variant_without_sku_anonymous(api_client, product, variant, channel_USD):
    if False:
        print('Hello World!')
    variant.sku = None
    variant.save()
    data = _fetch_variant(api_client, variant, channel_slug=channel_USD.slug)
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    product_id = graphene.Node.to_global_id('Product', product.pk)
    assert data['id'] == variant_id
    assert data['product']['id'] == product_id

def test_query_product_variant_for_federation(api_client, variant, channel_USD):
    if False:
        print('Hello World!')
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    variables = {'representations': [{'__typename': 'ProductVariant', 'id': variant_id, 'channel': channel_USD.slug}]}
    query = '\n      query GetProductVariantInFederation($representations: [_Any]) {\n        _entities(representations: $representations) {\n          __typename\n          ... on ProductVariant {\n            id\n            name\n          }\n        }\n      }\n    '
    response = api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    assert content['data']['_entities'] == [{'__typename': 'ProductVariant', 'id': variant_id, 'name': variant.name}]