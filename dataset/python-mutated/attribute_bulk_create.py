from ...utils import get_graphql_content
ATTRIBUTE_BULK_CREATE_MUTATION = '\nmutation AttributeBulkCreate($attributes: [AttributeCreateInput!]!) {\n  attributeBulkCreate(attributes: $attributes) {\n    results {\n      errors {\n        path\n        message\n        code\n      }\n      attribute {\n        id\n        name\n        slug\n        choices(first: 10) {\n          edges {\n            node {\n              id\n              name\n              slug\n              value\n              inputType\n              reference\n              file {\n                url\n                contentType\n              }\n              richText\n              plainText\n              boolean\n              date\n              dateTime\n            }\n          }\n        }\n      }\n    }\n    count\n  }\n}\n'

def bulk_create_attributes(staff_api_client, attributes=None):
    if False:
        for i in range(10):
            print('nop')
    variables = {'attributes': attributes}
    response = staff_api_client.post_graphql(ATTRIBUTE_BULK_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['attributeBulkCreate']['results'][0]['errors'] == []
    data = content['data']['attributeBulkCreate']
    return data