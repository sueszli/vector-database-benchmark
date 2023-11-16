from google.cloud import dialogflowcx_v3

def sample_create_entity_type():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.EntityTypesClient()
    entity_type = dialogflowcx_v3.EntityType()
    entity_type.display_name = 'display_name_value'
    entity_type.kind = 'KIND_REGEXP'
    request = dialogflowcx_v3.CreateEntityTypeRequest(parent='parent_value', entity_type=entity_type)
    response = client.create_entity_type(request=request)
    print(response)