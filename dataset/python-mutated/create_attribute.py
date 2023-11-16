from ...utils import get_graphql_content
ATTRIBUTE_CREATE_MUTATION = '\nmutation AttributeCreate($input: AttributeCreateInput!) {\n  attributeCreate(input: $input) {\n    attribute {\n      id\n    }\n    errors {\n      code\n      field\n      message\n    }\n  }\n}\n'

def attribute_create(staff_api_client, input_type='DROPDOWN', name='Color', slug='color', type='PRODUCT_TYPE', value_required=True, is_variant_only=False, values=None, unit=None, entityType=None):
    if False:
        i = 10
        return i + 15
    variables = {'input': {'inputType': input_type, 'name': name, 'slug': slug, 'type': type, 'valueRequired': value_required, 'isVariantOnly': is_variant_only, 'values': values, 'unit': unit, 'entityType': entityType}}
    response = staff_api_client.post_graphql(ATTRIBUTE_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['attributeCreate']['errors'] == []
    data = content['data']['attributeCreate']['attribute']
    assert data['id'] is not None
    return data