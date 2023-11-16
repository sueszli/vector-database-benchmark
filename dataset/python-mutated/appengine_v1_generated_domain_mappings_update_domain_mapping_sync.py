from google.cloud import appengine_admin_v1

def sample_update_domain_mapping():
    if False:
        return 10
    client = appengine_admin_v1.DomainMappingsClient()
    request = appengine_admin_v1.UpdateDomainMappingRequest()
    operation = client.update_domain_mapping(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)