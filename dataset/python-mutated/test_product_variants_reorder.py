import graphene
from .....graphql.tests.utils import get_graphql_content
from .....product.error_codes import ProductErrorCode
REORDER_PRODUCT_VARIANTS_MUTATION = '\n    mutation ProductVariantReorder($product: ID!, $moves: [ReorderInput!]!) {\n        productVariantReorder(productId: $product, moves: $moves) {\n            errors {\n                code\n                field\n            }\n            product {\n                id\n            }\n        }\n    }\n'

def test_reorder_variants(staff_api_client, product_with_two_variants, permission_manage_products):
    if False:
        print('Hello World!')
    default_variants = product_with_two_variants.variants.all()
    new_variants = [default_variants[1], default_variants[0]]
    variables = {'product': graphene.Node.to_global_id('Product', product_with_two_variants.pk), 'moves': [{'id': graphene.Node.to_global_id('ProductVariant', variant.pk), 'sortOrder': _order + 1} for (_order, variant) in enumerate(new_variants)]}
    response = staff_api_client.post_graphql(REORDER_PRODUCT_VARIANTS_MUTATION, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    data = content['data']['productVariantReorder']
    assert not data['errors']
    product_with_two_variants.refresh_from_db()
    assert list(product_with_two_variants.variants.all()) == new_variants

def test_reorder_variants_invalid_variants(staff_api_client, product, product_with_two_variants, permission_manage_products):
    if False:
        return 10
    default_variants = product_with_two_variants.variants.all()
    new_variants = [product.variants.first(), default_variants[1]]
    variables = {'product': graphene.Node.to_global_id('Product', product_with_two_variants.pk), 'moves': [{'id': graphene.Node.to_global_id('ProductVariant', variant.pk), 'sortOrder': _order + 1} for (_order, variant) in enumerate(new_variants)]}
    response = staff_api_client.post_graphql(REORDER_PRODUCT_VARIANTS_MUTATION, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    data = content['data']['productVariantReorder']
    assert data['errors'][0]['field'] == 'moves'
    assert data['errors'][0]['code'] == ProductErrorCode.NOT_FOUND.name