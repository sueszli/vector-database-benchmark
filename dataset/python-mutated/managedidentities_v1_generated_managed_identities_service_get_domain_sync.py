from google.cloud import managedidentities_v1

def sample_get_domain():
    if False:
        for i in range(10):
            print('nop')
    client = managedidentities_v1.ManagedIdentitiesServiceClient()
    request = managedidentities_v1.GetDomainRequest(name='name_value')
    response = client.get_domain(request=request)
    print(response)