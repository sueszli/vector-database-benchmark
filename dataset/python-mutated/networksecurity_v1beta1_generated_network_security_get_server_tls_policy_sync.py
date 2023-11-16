from google.cloud import network_security_v1beta1

def sample_get_server_tls_policy():
    if False:
        for i in range(10):
            print('nop')
    client = network_security_v1beta1.NetworkSecurityClient()
    request = network_security_v1beta1.GetServerTlsPolicyRequest(name='name_value')
    response = client.get_server_tls_policy(request=request)
    print(response)