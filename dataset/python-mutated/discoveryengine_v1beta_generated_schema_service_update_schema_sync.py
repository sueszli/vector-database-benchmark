from google.cloud import discoveryengine_v1beta

def sample_update_schema():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1beta.SchemaServiceClient()
    request = discoveryengine_v1beta.UpdateSchemaRequest()
    operation = client.update_schema(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)