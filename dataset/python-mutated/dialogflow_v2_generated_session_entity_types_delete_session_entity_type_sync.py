from google.cloud import dialogflow_v2

def sample_delete_session_entity_type():
    if False:
        return 10
    client = dialogflow_v2.SessionEntityTypesClient()
    request = dialogflow_v2.DeleteSessionEntityTypeRequest(name='name_value')
    client.delete_session_entity_type(request=request)