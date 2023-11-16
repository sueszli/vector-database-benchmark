from google.cloud import dialogflow_v2beta1

def sample_get_entity_type():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.EntityTypesClient()
    request = dialogflow_v2beta1.GetEntityTypeRequest(name='name_value')
    response = client.get_entity_type(request=request)
    print(response)