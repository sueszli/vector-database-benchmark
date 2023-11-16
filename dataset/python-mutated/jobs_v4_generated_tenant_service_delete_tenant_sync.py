from google.cloud import talent_v4

def sample_delete_tenant():
    if False:
        i = 10
        return i + 15
    client = talent_v4.TenantServiceClient()
    request = talent_v4.DeleteTenantRequest(name='name_value')
    client.delete_tenant(request=request)