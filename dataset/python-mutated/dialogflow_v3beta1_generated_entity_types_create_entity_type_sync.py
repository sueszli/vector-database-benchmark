from google.cloud import dialogflowcx_v3beta1

def sample_create_entity_type():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.EntityTypesClient()
    entity_type = dialogflowcx_v3beta1.EntityType()
    entity_type.display_name = 'display_name_value'
    entity_type.kind = 'KIND_REGEXP'
    request = dialogflowcx_v3beta1.CreateEntityTypeRequest(parent='parent_value', entity_type=entity_type)
    response = client.create_entity_type(request=request)
    print(response)