from ...utils import get_graphql_content
METADATA_UPDATE_MUTATION = '\nmutation UpdateMetadata(\n    $id: ID!,\n    $input:[MetadataInput!]!,\n   ) {\n  updateMetadata(id: $id, input: $input) {\n    errors {\n        message\n        field\n    }\n    item {\n      metadata {\n        key\n        value\n      }\n    }\n  }\n}\n\n'

def update_metadata(staff_api_client, id, input):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': id, 'input': input}
    response = staff_api_client.post_graphql(METADATA_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['updateMetadata']['errors'] == []
    metadata = content['data']['updateMetadata']['item']['metadata']
    assert metadata is not None
    return metadata