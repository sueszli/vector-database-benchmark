from google.cloud import dialogflowcx_v3

def sample_get_session_entity_type():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.SessionEntityTypesClient()
    request = dialogflowcx_v3.GetSessionEntityTypeRequest(name='name_value')
    response = client.get_session_entity_type(request=request)
    print(response)