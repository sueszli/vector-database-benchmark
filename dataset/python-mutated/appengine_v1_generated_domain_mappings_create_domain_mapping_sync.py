from google.cloud import appengine_admin_v1

def sample_create_domain_mapping():
    if False:
        while True:
            i = 10
    client = appengine_admin_v1.DomainMappingsClient()
    request = appengine_admin_v1.CreateDomainMappingRequest()
    operation = client.create_domain_mapping(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)