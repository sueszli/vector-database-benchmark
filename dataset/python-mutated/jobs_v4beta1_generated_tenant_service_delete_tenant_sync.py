from google.cloud import talent_v4beta1

def sample_delete_tenant():
    if False:
        return 10
    client = talent_v4beta1.TenantServiceClient()
    request = talent_v4beta1.DeleteTenantRequest(name='name_value')
    client.delete_tenant(request=request)