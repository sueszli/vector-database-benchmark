from google.cloud import dialogflow_v2

def sample_get_entity_type():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.EntityTypesClient()
    request = dialogflow_v2.GetEntityTypeRequest(name='name_value')
    response = client.get_entity_type(request=request)
    print(response)