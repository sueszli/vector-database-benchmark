from google.cloud import network_security_v1

def sample_get_authorization_policy():
    if False:
        i = 10
        return i + 15
    client = network_security_v1.NetworkSecurityClient()
    request = network_security_v1.GetAuthorizationPolicyRequest(name='name_value')
    response = client.get_authorization_policy(request=request)
    print(response)