from google.cloud import talent_v4beta1

def sample_get_tenant():
    if False:
        for i in range(10):
            print('nop')
    client = talent_v4beta1.TenantServiceClient()
    request = talent_v4beta1.GetTenantRequest(name='name_value')
    response = client.get_tenant(request=request)
    print(response)