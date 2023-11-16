from google.cloud import apigee_registry_v1

def sample_rollback_api_spec():
    if False:
        while True:
            i = 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.RollbackApiSpecRequest(name='name_value', revision_id='revision_id_value')
    response = client.rollback_api_spec(request=request)
    print(response)