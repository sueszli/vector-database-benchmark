from google.cloud import dialogflowcx_v3

def sample_list_entity_types():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.EntityTypesClient()
    request = dialogflowcx_v3.ListEntityTypesRequest(parent='parent_value')
    page_result = client.list_entity_types(request=request)
    for response in page_result:
        print(response)