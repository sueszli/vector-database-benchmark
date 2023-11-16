from google.cloud import discoveryengine_v1beta

def sample_delete_schema():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1beta.SchemaServiceClient()
    request = discoveryengine_v1beta.DeleteSchemaRequest(name='name_value')
    operation = client.delete_schema(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)