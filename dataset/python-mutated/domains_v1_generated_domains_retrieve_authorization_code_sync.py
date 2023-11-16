from google.cloud import domains_v1

def sample_retrieve_authorization_code():
    if False:
        i = 10
        return i + 15
    client = domains_v1.DomainsClient()
    request = domains_v1.RetrieveAuthorizationCodeRequest(registration='registration_value')
    response = client.retrieve_authorization_code(request=request)
    print(response)