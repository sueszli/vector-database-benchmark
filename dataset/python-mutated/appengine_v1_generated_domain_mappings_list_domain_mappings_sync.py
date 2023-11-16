from google.cloud import appengine_admin_v1

def sample_list_domain_mappings():
    if False:
        i = 10
        return i + 15
    client = appengine_admin_v1.DomainMappingsClient()
    request = appengine_admin_v1.ListDomainMappingsRequest()
    page_result = client.list_domain_mappings(request=request)
    for response in page_result:
        print(response)