from google.cloud import appengine_admin_v1

def sample_get_domain_mapping():
    if False:
        return 10
    client = appengine_admin_v1.DomainMappingsClient()
    request = appengine_admin_v1.GetDomainMappingRequest()
    response = client.get_domain_mapping(request=request)
    print(response)