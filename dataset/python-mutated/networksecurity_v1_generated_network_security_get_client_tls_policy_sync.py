from google.cloud import network_security_v1

def sample_get_client_tls_policy():
    if False:
        return 10
    client = network_security_v1.NetworkSecurityClient()
    request = network_security_v1.GetClientTlsPolicyRequest(name='name_value')
    response = client.get_client_tls_policy(request=request)
    print(response)