from google.cloud import domains_v1beta1

def sample_update_registration():
    if False:
        while True:
            i = 10
    client = domains_v1beta1.DomainsClient()
    request = domains_v1beta1.UpdateRegistrationRequest()
    operation = client.update_registration(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)