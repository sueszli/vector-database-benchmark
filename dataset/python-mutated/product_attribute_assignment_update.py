from ...utils import get_graphql_content
PRODUCT_ATTRIBUTE_ASSIGNMENT_UPDATE_MUTATION = '\nmutation ProductAttributeAssignmentUpdate(\n    $operations: [ProductAttributeAssignmentUpdateInput!]!, $id: ID!) {\n  productAttributeAssignmentUpdate(operations: $operations, productTypeId: $id) {\n    errors {\n      field\n      code\n      message\n    }\n    productType {\n      id\n      hasVariants\n      productAttributes {\n        id\n      }\n      assignedVariantAttributes {\n        attribute {\n          id\n        }\n        variantSelection\n      }\n    }\n  }\n}\n'

def update_product_type_assignment_attribute(staff_api_client, product_type_id, operations):
    if False:
        print('Hello World!')
    variables = {'id': product_type_id, 'operations': operations}
    response = staff_api_client.post_graphql(PRODUCT_ATTRIBUTE_ASSIGNMENT_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['productAttributeAssignmentUpdate']['errors'] == []
    data = content['data']['productAttributeAssignmentUpdate']['productType']
    assert data['id'] is not None
    return data