from google.cloud import discoveryengine_v1beta

def sample_create_schema():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1beta.SchemaServiceClient()
    request = discoveryengine_v1beta.CreateSchemaRequest(parent='parent_value', schema_id='schema_id_value')
    operation = client.create_schema(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)