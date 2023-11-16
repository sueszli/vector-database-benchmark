from ...utils import get_graphql_content
PRIVATE_METADATA_UPDATE_MUTATION = '\nmutation UpdatePrivateMetadata(\n    $id: ID!,\n    $input:[MetadataInput!]!,\n   ) {\n  updatePrivateMetadata(id: $id, input: $input) {\n    errors {\n        message\n        field\n    }\n    item {\n        privateMetadata {\n            key\n            value\n        }\n    }\n  }\n}\n\n'

def update_private_metadata(staff_api_client, id, input):
    if False:
        print('Hello World!')
    variables = {'id': id, 'input': input}
    response = staff_api_client.post_graphql(PRIVATE_METADATA_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['updatePrivateMetadata']['errors'] == []
    data = content['data']['updatePrivateMetadata']['item']['privateMetadata']
    assert data is not None
    return data