from google.cloud import dialogflow_v2beta1

def sample_create_entity_type():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.EntityTypesClient()
    entity_type = dialogflow_v2beta1.EntityType()
    entity_type.display_name = 'display_name_value'
    entity_type.kind = 'KIND_REGEXP'
    request = dialogflow_v2beta1.CreateEntityTypeRequest(parent='parent_value', entity_type=entity_type)
    response = client.create_entity_type(request=request)
    print(response)