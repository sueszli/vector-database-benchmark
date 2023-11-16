from google.cloud import dialogflow_v2beta1

def sample_delete_session_entity_type():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.SessionEntityTypesClient()
    request = dialogflow_v2beta1.DeleteSessionEntityTypeRequest(name='name_value')
    client.delete_session_entity_type(request=request)