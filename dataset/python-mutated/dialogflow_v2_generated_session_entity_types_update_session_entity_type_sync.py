from google.cloud import dialogflow_v2

def sample_update_session_entity_type():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.SessionEntityTypesClient()
    session_entity_type = dialogflow_v2.SessionEntityType()
    session_entity_type.name = 'name_value'
    session_entity_type.entity_override_mode = 'ENTITY_OVERRIDE_MODE_SUPPLEMENT'
    session_entity_type.entities.value = 'value_value'
    session_entity_type.entities.synonyms = ['synonyms_value1', 'synonyms_value2']
    request = dialogflow_v2.UpdateSessionEntityTypeRequest(session_entity_type=session_entity_type)
    response = client.update_session_entity_type(request=request)
    print(response)