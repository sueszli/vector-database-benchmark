from google.cloud import dialogflowcx_v3beta1

def sample_delete_entity_type():
    if False:
        return 10
    client = dialogflowcx_v3beta1.EntityTypesClient()
    request = dialogflowcx_v3beta1.DeleteEntityTypeRequest(name='name_value')
    client.delete_entity_type(request=request)