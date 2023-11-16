from google.cloud import discoveryengine_v1

def sample_list_schemas():
    if False:
        return 10
    client = discoveryengine_v1.SchemaServiceClient()
    request = discoveryengine_v1.ListSchemasRequest(parent='parent_value')
    page_result = client.list_schemas(request=request)
    for response in page_result:
        print(response)