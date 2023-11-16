from google.cloud import discoveryengine_v1alpha

def sample_get_schema():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1alpha.SchemaServiceClient()
    request = discoveryengine_v1alpha.GetSchemaRequest(name='name_value')
    response = client.get_schema(request=request)
    print(response)