from google.cloud import domains_v1

def sample_delete_registration():
    if False:
        while True:
            i = 10
    client = domains_v1.DomainsClient()
    request = domains_v1.DeleteRegistrationRequest(name='name_value')
    operation = client.delete_registration(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)