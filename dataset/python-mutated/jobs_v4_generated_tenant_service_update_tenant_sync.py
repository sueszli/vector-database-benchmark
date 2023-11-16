from google.cloud import talent_v4

def sample_update_tenant():
    if False:
        return 10
    client = talent_v4.TenantServiceClient()
    tenant = talent_v4.Tenant()
    tenant.external_id = 'external_id_value'
    request = talent_v4.UpdateTenantRequest(tenant=tenant)
    response = client.update_tenant(request=request)
    print(response)