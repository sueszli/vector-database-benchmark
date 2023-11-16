from google.cloud import dialogflow_v2

def sample_get_session_entity_type():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.SessionEntityTypesClient()
    request = dialogflow_v2.GetSessionEntityTypeRequest(name='name_value')
    response = client.get_session_entity_type(request=request)
    print(response)