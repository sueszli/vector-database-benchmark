from google.cloud import talent_v4

def sample_create_tenant():
    if False:
        while True:
            i = 10
    client = talent_v4.TenantServiceClient()
    tenant = talent_v4.Tenant()
    tenant.external_id = 'external_id_value'
    request = talent_v4.CreateTenantRequest(parent='parent_value', tenant=tenant)
    response = client.create_tenant(request=request)
    print(response)