from google.cloud import discoveryengine_v1

def sample_delete_schema():
    if False:
        print('Hello World!')
    client = discoveryengine_v1.SchemaServiceClient()
    request = discoveryengine_v1.DeleteSchemaRequest(name='name_value')
    operation = client.delete_schema(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)