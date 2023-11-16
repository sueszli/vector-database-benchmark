from google.cloud import talent_v4

def sample_list_tenants():
    if False:
        i = 10
        return i + 15
    client = talent_v4.TenantServiceClient()
    request = talent_v4.ListTenantsRequest(parent='parent_value')
    page_result = client.list_tenants(request=request)
    for response in page_result:
        print(response)