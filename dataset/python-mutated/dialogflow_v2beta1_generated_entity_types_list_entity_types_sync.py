from google.cloud import dialogflow_v2beta1

def sample_list_entity_types():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.EntityTypesClient()
    request = dialogflow_v2beta1.ListEntityTypesRequest(parent='parent_value')
    page_result = client.list_entity_types(request=request)
    for response in page_result:
        print(response)