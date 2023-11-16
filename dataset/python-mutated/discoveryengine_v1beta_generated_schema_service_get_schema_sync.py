from google.cloud import discoveryengine_v1beta

def sample_get_schema():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1beta.SchemaServiceClient()
    request = discoveryengine_v1beta.GetSchemaRequest(name='name_value')
    response = client.get_schema(request=request)
    print(response)