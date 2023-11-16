from google.cloud import domains_v1beta1

def sample_delete_registration():
    if False:
        i = 10
        return i + 15
    client = domains_v1beta1.DomainsClient()
    request = domains_v1beta1.DeleteRegistrationRequest(name='name_value')
    operation = client.delete_registration(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)