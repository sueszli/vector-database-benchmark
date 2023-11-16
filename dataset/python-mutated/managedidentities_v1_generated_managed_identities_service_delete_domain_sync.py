from google.cloud import managedidentities_v1

def sample_delete_domain():
    if False:
        i = 10
        return i + 15
    client = managedidentities_v1.ManagedIdentitiesServiceClient()
    request = managedidentities_v1.DeleteDomainRequest(name='name_value')
    operation = client.delete_domain(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)