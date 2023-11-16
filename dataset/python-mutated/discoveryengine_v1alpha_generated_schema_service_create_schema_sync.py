from google.cloud import discoveryengine_v1alpha

def sample_create_schema():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1alpha.SchemaServiceClient()
    request = discoveryengine_v1alpha.CreateSchemaRequest(parent='parent_value', schema_id='schema_id_value')
    operation = client.create_schema(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)