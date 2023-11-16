from google.cloud import dialogflow_v2

def sample_list_entity_types():
    if False:
        return 10
    client = dialogflow_v2.EntityTypesClient()
    request = dialogflow_v2.ListEntityTypesRequest(parent='parent_value')
    page_result = client.list_entity_types(request=request)
    for response in page_result:
        print(response)