from google.cloud import discoveryengine_v1alpha

def sample_delete_schema():
    if False:
        return 10
    client = discoveryengine_v1alpha.SchemaServiceClient()
    request = discoveryengine_v1alpha.DeleteSchemaRequest(name='name_value')
    operation = client.delete_schema(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)