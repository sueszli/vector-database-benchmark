from google.cloud import dialogflowcx_v3beta1

def sample_list_entity_types():
    if False:
        return 10
    client = dialogflowcx_v3beta1.EntityTypesClient()
    request = dialogflowcx_v3beta1.ListEntityTypesRequest(parent='parent_value')
    page_result = client.list_entity_types(request=request)
    for response in page_result:
        print(response)