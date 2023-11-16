from google.cloud import dialogflowcx_v3beta1

def sample_get_session_entity_type():
    if False:
        return 10
    client = dialogflowcx_v3beta1.SessionEntityTypesClient()
    request = dialogflowcx_v3beta1.GetSessionEntityTypeRequest(name='name_value')
    response = client.get_session_entity_type(request=request)
    print(response)