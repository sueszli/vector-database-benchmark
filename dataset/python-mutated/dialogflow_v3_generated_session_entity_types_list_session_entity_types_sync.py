from google.cloud import dialogflowcx_v3

def sample_list_session_entity_types():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.SessionEntityTypesClient()
    request = dialogflowcx_v3.ListSessionEntityTypesRequest(parent='parent_value')
    page_result = client.list_session_entity_types(request=request)
    for response in page_result:
        print(response)