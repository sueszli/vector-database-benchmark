from google.cloud import dialogflow_v2beta1

def sample_create_session_entity_type():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.SessionEntityTypesClient()
    request = dialogflow_v2beta1.CreateSessionEntityTypeRequest(parent='parent_value')
    response = client.create_session_entity_type(request=request)
    print(response)