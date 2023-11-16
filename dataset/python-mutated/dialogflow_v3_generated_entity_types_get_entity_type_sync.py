from google.cloud import dialogflowcx_v3

def sample_get_entity_type():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.EntityTypesClient()
    request = dialogflowcx_v3.GetEntityTypeRequest(name='name_value')
    response = client.get_entity_type(request=request)
    print(response)