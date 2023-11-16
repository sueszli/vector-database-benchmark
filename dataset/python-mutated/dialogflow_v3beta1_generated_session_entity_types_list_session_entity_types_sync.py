from google.cloud import dialogflowcx_v3beta1

def sample_list_session_entity_types():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.SessionEntityTypesClient()
    request = dialogflowcx_v3beta1.ListSessionEntityTypesRequest(parent='parent_value')
    page_result = client.list_session_entity_types(request=request)
    for response in page_result:
        print(response)