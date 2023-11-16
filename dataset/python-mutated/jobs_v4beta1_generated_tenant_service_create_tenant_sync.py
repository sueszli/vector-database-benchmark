from google.cloud import talent_v4beta1

def sample_create_tenant():
    if False:
        for i in range(10):
            print('nop')
    client = talent_v4beta1.TenantServiceClient()
    tenant = talent_v4beta1.Tenant()
    tenant.external_id = 'external_id_value'
    request = talent_v4beta1.CreateTenantRequest(parent='parent_value', tenant=tenant)
    response = client.create_tenant(request=request)
    print(response)