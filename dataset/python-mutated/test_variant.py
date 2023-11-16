from uuid import uuid4
import graphene
import pytest
from .....product.models import ProductMedia, ProductVariant, VariantMedia
from .....warehouse.models import Stock
from ....tests.utils import get_graphql_content

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_retrieve_variant_list(product_variant_list, api_client, count_queries, warehouse, warehouse_no_shipping_zone, shipping_zone_without_countries, channel_USD):
    if False:
        print('Hello World!')
    query = '\n        fragment BasicProductFields on Product {\n          id\n          name\n          thumbnail {\n            url\n            alt\n          }\n          thumbnail2x: thumbnail(size: 510) {\n            url\n          }\n        }\n\n        fragment ProductVariantFields on ProductVariant {\n          id\n          sku\n          name\n          pricing {\n            discountLocalCurrency {\n              currency\n              gross {\n                amount\n              }\n            }\n            price {\n              currency\n              gross {\n                amount\n              }\n            }\n            priceUndiscounted {\n              currency\n              gross {\n                amount\n              }\n            }\n            priceLocalCurrency {\n              currency\n              gross {\n                amount\n              }\n            }\n          }\n          attributes {\n            attribute {\n              id\n              name\n            }\n            values {\n              id\n              name\n              value: name\n            }\n          }\n        }\n\n        query VariantList($ids: [ID!], $channel: String) {\n          productVariants(ids: $ids, first: 100, channel: $channel) {\n            edges {\n              node {\n                ...ProductVariantFields\n                quantityAvailable\n                quantityAvailablePl: quantityAvailable(countryCode: PL)\n                quantityAvailableUS: quantityAvailable(countryCode: US)\n                product {\n                  ...BasicProductFields\n                }\n              }\n            }\n          }\n        }\n    '
    warehouse_2 = warehouse_no_shipping_zone
    warehouse_2.shipping_zones.add(shipping_zone_without_countries)
    stocks = [Stock(product_variant=variant, warehouse=warehouse, quantity=1) for variant in product_variant_list]
    stocks.extend([Stock(product_variant=variant, warehouse=warehouse_2, quantity=2) for variant in product_variant_list])
    Stock.objects.bulk_create(stocks)
    variables = {'ids': [graphene.Node.to_global_id('ProductVariant', variant.pk) for variant in product_variant_list], 'channel': channel_USD.slug}
    get_graphql_content(api_client.post_graphql(query, variables))

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_product_variant_bulk_create(staff_api_client, product_with_variant_with_two_attributes, permission_manage_products, color_attribute, size_attribute, count_queries):
    if False:
        for i in range(10):
            print('nop')
    query = '\n    mutation ProductVariantBulkCreate(\n        $variants: [ProductVariantBulkCreateInput!]!, $productId: ID!\n    ) {\n        productVariantBulkCreate(variants: $variants, product: $productId) {\n            errors {\n                field\n                message\n                code\n                index\n            }\n            productVariants{\n                id\n                sku\n            }\n            count\n        }\n    }\n    '
    product = product_with_variant_with_two_attributes
    product_variant_count = ProductVariant.objects.count()
    product_id = graphene.Node.to_global_id('Product', product.pk)
    color_attribute_id = graphene.Node.to_global_id('Attribute', color_attribute.id)
    size_attribute_id = graphene.Node.to_global_id('Attribute', size_attribute.id)
    variants = [{'sku': str(uuid4())[:12], 'attributes': [{'id': color_attribute_id, 'values': ['red']}, {'id': size_attribute_id, 'values': ['big']}]}]
    variables = {'productId': product_id, 'variants': variants}
    staff_api_client.user.user_permissions.add(permission_manage_products)
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['productVariantBulkCreate']
    assert not data['errors']
    assert data['count'] == 1
    assert product_variant_count + 1 == ProductVariant.objects.count()

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_product_variant_create(staff_api_client, permission_manage_products, product_type, product_available_in_many_channels, warehouse, settings, count_queries):
    if False:
        print('Hello World!')
    query = '\n        mutation createVariant (\n            $productId: ID!,\n            $sku: String,\n            $stocks: [StockInput!],\n            $attributes: [AttributeValueInput!]!,\n            $weight: WeightScalar,\n            $trackInventory: Boolean\n        ) {\n            productVariantCreate(\n                input: {\n                    product: $productId,\n                    sku: $sku,\n                    stocks: $stocks,\n                    attributes: $attributes,\n                    trackInventory: $trackInventory,\n                    weight: $weight\n                }\n            ) {\n                errors {\n                    field\n                    message\n                    attributes\n                    code\n                }\n                productVariant {\n                    id\n                    name\n                    sku\n                    attributes {\n                        attribute {\n                            slug\n                        }\n                        values {\n                            name\n                            slug\n                            reference\n                            file {\n                                url\n                                contentType\n                            }\n                        }\n                    }\n                    weight {\n                        value\n                        unit\n                    }\n                    stocks {\n                        quantity\n                    warehouse {\n                        slug\n                    }\n                }\n            }\n        }\n    }\n    '
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    product = product_available_in_many_channels
    product_id = graphene.Node.to_global_id('Product', product.pk)
    sku = '1'
    weight = 10.22
    attribute_id = graphene.Node.to_global_id('Attribute', product_type.variant_attributes.first().pk)
    stocks = [{'warehouse': graphene.Node.to_global_id('Warehouse', warehouse.pk), 'quantity': 20}]
    variables = {'productId': product_id, 'sku': sku, 'stocks': stocks, 'weight': weight, 'attributes': [{'id': attribute_id, 'values': ['red']}], 'trackInventory': True}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)['data']['productVariantCreate']
    assert not content['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_update_product_variant(staff_api_client, permission_manage_products, product_available_in_many_channels, product_type, media_root, settings, image, count_queries):
    if False:
        for i in range(10):
            print('nop')
    query = '\n        mutation VariantUpdate(\n            $id: ID!\n            $attributes: [AttributeValueInput!]\n            $sku: String\n            $trackInventory: Boolean!\n        ) {\n            productVariantUpdate(\n                id: $id\n                input: {\n                    attributes: $attributes\n                    sku: $sku\n                    trackInventory: $trackInventory\n                }\n            ) {\n            errors {\n                field\n                message\n            }\n            productVariant {\n                id\n                attributes {\n                    attribute {\n                        id\n                        name\n                        slug\n                        choices(first: 10) {\n                            edges {\n                                node {\n                                    id\n                                    name\n                                    slug\n                                    __typename\n                                }\n                            }\n                        }\n                    __typename\n                    }\n                __typename\n                }\n            }\n        }\n    }\n    '
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    product = product_available_in_many_channels
    variant = product.variants.first()
    product_image = ProductMedia.objects.create(product=product, image=image)
    VariantMedia.objects.create(variant=variant, media=product_image)
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    attribute_id = graphene.Node.to_global_id('Attribute', product_type.variant_attributes.first().pk)
    variables = {'attributes': [{'id': attribute_id, 'values': ['yellow']}], 'id': variant_id, 'sku': '21599567', 'trackInventory': True}
    data = get_graphql_content(staff_api_client.post_graphql(query, variables, permissions=[permission_manage_products]))['data']['productVariantUpdate']
    assert not data['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_products_variants_for_federation_query_count(api_client, product_variant_list, channel_USD, django_assert_num_queries, count_queries):
    if False:
        while True:
            i = 10
    query = '\n      query GetProductVariantInFederation($representations: [_Any]) {\n        _entities(representations: $representations) {\n          __typename\n          ... on ProductVariant {\n            id\n            name\n          }\n        }\n      }\n    '
    variables = {'representations': [{'__typename': 'ProductVariant', 'id': graphene.Node.to_global_id('ProductVariant', product_variant_list[0].pk), 'channel': channel_USD.slug}]}
    with django_assert_num_queries(3):
        response = api_client.post_graphql(query, variables)
        content = get_graphql_content(response)
        assert len(content['data']['_entities']) == 1
    variables = {'representations': [{'__typename': 'ProductVariant', 'id': graphene.Node.to_global_id('ProductVariant', variant.pk), 'channel': channel_USD.slug} for variant in product_variant_list]}
    with django_assert_num_queries(3):
        response = api_client.post_graphql(query, variables)
        content = get_graphql_content(response)
        assert len(content['data']['_entities']) == 4