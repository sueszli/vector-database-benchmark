from google.cloud import dialogflow_v2

def sample_update_entity_type():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.EntityTypesClient()
    entity_type = dialogflow_v2.EntityType()
    entity_type.display_name = 'display_name_value'
    entity_type.kind = 'KIND_REGEXP'
    request = dialogflow_v2.UpdateEntityTypeRequest(entity_type=entity_type)
    response = client.update_entity_type(request=request)
    print(response)