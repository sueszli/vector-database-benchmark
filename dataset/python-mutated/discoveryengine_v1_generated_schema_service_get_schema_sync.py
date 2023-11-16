from google.cloud import discoveryengine_v1

def sample_get_schema():
    if False:
        print('Hello World!')
    client = discoveryengine_v1.SchemaServiceClient()
    request = discoveryengine_v1.GetSchemaRequest(name='name_value')
    response = client.get_schema(request=request)
    print(response)