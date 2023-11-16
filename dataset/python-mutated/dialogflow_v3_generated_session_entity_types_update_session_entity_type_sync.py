from google.cloud import dialogflowcx_v3

def sample_update_session_entity_type():
    if False:
        return 10
    client = dialogflowcx_v3.SessionEntityTypesClient()
    session_entity_type = dialogflowcx_v3.SessionEntityType()
    session_entity_type.name = 'name_value'
    session_entity_type.entity_override_mode = 'ENTITY_OVERRIDE_MODE_SUPPLEMENT'
    session_entity_type.entities.value = 'value_value'
    session_entity_type.entities.synonyms = ['synonyms_value1', 'synonyms_value2']
    request = dialogflowcx_v3.UpdateSessionEntityTypeRequest(session_entity_type=session_entity_type)
    response = client.update_session_entity_type(request=request)
    print(response)