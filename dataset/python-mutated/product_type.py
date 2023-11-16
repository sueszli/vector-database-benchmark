from ...utils import get_graphql_content
PRODUCT_TYPE_CREATE_MUTATION = '\nmutation createProductType($input: ProductTypeInput!) {\n  productTypeCreate(input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    productType {\n      id\n      name\n      slug\n      kind\n      isShippingRequired\n      isDigital\n      hasVariants\n      productAttributes {\n        id\n      }\n      assignedVariantAttributes {\n        attribute {\n          id\n        }\n        variantSelection\n      }\n    }\n  }\n}\n'

def create_product_type(staff_api_client, product_type_name='Test type', slug='test-type', is_shipping_required=True, is_digital=False, has_variants=False, product_attributes=None, variant_attributes=None, kind='NORMAL'):
    if False:
        while True:
            i = 10
    if not product_attributes:
        product_attributes = []
    if not variant_attributes:
        variant_attributes = []
    variables = {'input': {'name': product_type_name, 'slug': slug, 'isShippingRequired': is_shipping_required, 'isDigital': is_digital, 'hasVariants': has_variants, 'productAttributes': product_attributes, 'variantAttributes': variant_attributes, 'kind': kind}}
    response = staff_api_client.post_graphql(PRODUCT_TYPE_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['productTypeCreate']['errors'] == []
    data = content['data']['productTypeCreate']['productType']
    assert data['id'] is not None
    assert data['name'] == product_type_name
    assert data['slug'] == slug
    assert data['isShippingRequired'] is is_shipping_required
    assert data['isDigital'] is is_digital
    return data