from google.cloud import network_security_v1

def sample_get_server_tls_policy():
    if False:
        while True:
            i = 10
    client = network_security_v1.NetworkSecurityClient()
    request = network_security_v1.GetServerTlsPolicyRequest(name='name_value')
    response = client.get_server_tls_policy(request=request)
    print(response)