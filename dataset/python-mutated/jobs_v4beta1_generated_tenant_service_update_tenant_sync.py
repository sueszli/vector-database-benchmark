from google.cloud import talent_v4beta1

def sample_update_tenant():
    if False:
        for i in range(10):
            print('nop')
    client = talent_v4beta1.TenantServiceClient()
    tenant = talent_v4beta1.Tenant()
    tenant.external_id = 'external_id_value'
    request = talent_v4beta1.UpdateTenantRequest(tenant=tenant)
    response = client.update_tenant(request=request)
    print(response)