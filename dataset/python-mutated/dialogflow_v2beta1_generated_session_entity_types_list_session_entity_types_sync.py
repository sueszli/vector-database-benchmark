from google.cloud import dialogflow_v2beta1

def sample_list_session_entity_types():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.SessionEntityTypesClient()
    request = dialogflow_v2beta1.ListSessionEntityTypesRequest(parent='parent_value')
    page_result = client.list_session_entity_types(request=request)
    for response in page_result:
        print(response)