from google.cloud import domains_v1

def sample_update_registration():
    if False:
        print('Hello World!')
    client = domains_v1.DomainsClient()
    request = domains_v1.UpdateRegistrationRequest()
    operation = client.update_registration(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)