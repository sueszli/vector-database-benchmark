from google.cloud import dialogflowcx_v3

def sample_delete_session_entity_type():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.SessionEntityTypesClient()
    request = dialogflowcx_v3.DeleteSessionEntityTypeRequest(name='name_value')
    client.delete_session_entity_type(request=request)