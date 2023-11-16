from google.cloud import dialogflow_v2beta1

def sample_get_session_entity_type():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2beta1.SessionEntityTypesClient()
    request = dialogflow_v2beta1.GetSessionEntityTypeRequest(name='name_value')
    response = client.get_session_entity_type(request=request)
    print(response)