from google.cloud import dialogflow_v2

def sample_delete_entity_type():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.EntityTypesClient()
    request = dialogflow_v2.DeleteEntityTypeRequest(name='name_value')
    client.delete_entity_type(request=request)