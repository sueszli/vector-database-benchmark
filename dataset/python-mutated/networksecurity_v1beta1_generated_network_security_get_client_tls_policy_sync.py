from google.cloud import network_security_v1beta1

def sample_get_client_tls_policy():
    if False:
        while True:
            i = 10
    client = network_security_v1beta1.NetworkSecurityClient()
    request = network_security_v1beta1.GetClientTlsPolicyRequest(name='name_value')
    response = client.get_client_tls_policy(request=request)
    print(response)