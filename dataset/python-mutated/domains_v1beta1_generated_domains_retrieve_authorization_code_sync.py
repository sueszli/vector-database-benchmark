from google.cloud import domains_v1beta1

def sample_retrieve_authorization_code():
    if False:
        while True:
            i = 10
    client = domains_v1beta1.DomainsClient()
    request = domains_v1beta1.RetrieveAuthorizationCodeRequest(registration='registration_value')
    response = client.retrieve_authorization_code(request=request)
    print(response)