from google.cloud import dialogflowcx_v3

def sample_delete_entity_type():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.EntityTypesClient()
    request = dialogflowcx_v3.DeleteEntityTypeRequest(name='name_value')
    client.delete_entity_type(request=request)