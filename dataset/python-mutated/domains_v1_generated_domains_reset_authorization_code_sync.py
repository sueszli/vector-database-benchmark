from google.cloud import domains_v1

def sample_reset_authorization_code():
    if False:
        print('Hello World!')
    client = domains_v1.DomainsClient()
    request = domains_v1.ResetAuthorizationCodeRequest(registration='registration_value')
    response = client.reset_authorization_code(request=request)
    print(response)