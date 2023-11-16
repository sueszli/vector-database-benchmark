from google.cloud import discoveryengine_v1alpha

def sample_list_schemas():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1alpha.SchemaServiceClient()
    request = discoveryengine_v1alpha.ListSchemasRequest(parent='parent_value')
    page_result = client.list_schemas(request=request)
    for response in page_result:
        print(response)