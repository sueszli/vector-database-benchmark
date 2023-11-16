from google.cloud import dialogflow_v2beta1

def sample_delete_entity_type():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.EntityTypesClient()
    request = dialogflow_v2beta1.DeleteEntityTypeRequest(name='name_value')
    client.delete_entity_type(request=request)