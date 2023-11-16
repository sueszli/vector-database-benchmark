from google.cloud import dialogflowcx_v3beta1

def sample_get_entity_type():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3beta1.EntityTypesClient()
    request = dialogflowcx_v3beta1.GetEntityTypeRequest(name='name_value')
    response = client.get_entity_type(request=request)
    print(response)