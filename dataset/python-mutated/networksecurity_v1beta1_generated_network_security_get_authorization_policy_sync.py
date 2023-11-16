from google.cloud import network_security_v1beta1

def sample_get_authorization_policy():
    if False:
        return 10
    client = network_security_v1beta1.NetworkSecurityClient()
    request = network_security_v1beta1.GetAuthorizationPolicyRequest(name='name_value')
    response = client.get_authorization_policy(request=request)
    print(response)