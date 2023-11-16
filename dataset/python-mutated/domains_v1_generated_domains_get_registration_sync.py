from google.cloud import domains_v1

def sample_get_registration():
    if False:
        print('Hello World!')
    client = domains_v1.DomainsClient()
    request = domains_v1.GetRegistrationRequest(name='name_value')
    response = client.get_registration(request=request)
    print(response)