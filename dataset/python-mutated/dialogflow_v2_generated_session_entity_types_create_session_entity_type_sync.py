from google.cloud import dialogflow_v2

def sample_create_session_entity_type():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.SessionEntityTypesClient()
    session_entity_type = dialogflow_v2.SessionEntityType()
    session_entity_type.name = 'name_value'
    session_entity_type.entity_override_mode = 'ENTITY_OVERRIDE_MODE_SUPPLEMENT'
    session_entity_type.entities.value = 'value_value'
    session_entity_type.entities.synonyms = ['synonyms_value1', 'synonyms_value2']
    request = dialogflow_v2.CreateSessionEntityTypeRequest(parent='parent_value', session_entity_type=session_entity_type)
    response = client.create_session_entity_type(request=request)
    print(response)