from google.cloud import discoveryengine_v1

def sample_update_schema():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1.SchemaServiceClient()
    request = discoveryengine_v1.UpdateSchemaRequest()
    operation = client.update_schema(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)