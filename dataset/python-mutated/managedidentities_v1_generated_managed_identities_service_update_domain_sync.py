from google.cloud import managedidentities_v1

def sample_update_domain():
    if False:
        return 10
    client = managedidentities_v1.ManagedIdentitiesServiceClient()
    domain = managedidentities_v1.Domain()
    domain.name = 'name_value'
    domain.reserved_ip_range = 'reserved_ip_range_value'
    domain.locations = ['locations_value1', 'locations_value2']
    request = managedidentities_v1.UpdateDomainRequest(domain=domain)
    operation = client.update_domain(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)