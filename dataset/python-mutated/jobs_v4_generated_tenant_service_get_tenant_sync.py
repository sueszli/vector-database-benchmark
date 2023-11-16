from google.cloud import talent_v4

def sample_get_tenant():
    if False:
        while True:
            i = 10
    client = talent_v4.TenantServiceClient()
    request = talent_v4.GetTenantRequest(name='name_value')
    response = client.get_tenant(request=request)
    print(response)