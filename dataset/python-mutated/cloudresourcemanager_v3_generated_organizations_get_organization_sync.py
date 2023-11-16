from google.cloud import resourcemanager_v3

def sample_get_organization():
    if False:
        for i in range(10):
            print('nop')
    client = resourcemanager_v3.OrganizationsClient()
    request = resourcemanager_v3.GetOrganizationRequest(name='name_value')
    response = client.get_organization(request=request)
    print(response)