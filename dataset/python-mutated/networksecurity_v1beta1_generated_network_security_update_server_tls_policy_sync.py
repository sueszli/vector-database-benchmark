from google.cloud import network_security_v1beta1

def sample_update_server_tls_policy():
    if False:
        i = 10
        return i + 15
    client = network_security_v1beta1.NetworkSecurityClient()
    server_tls_policy = network_security_v1beta1.ServerTlsPolicy()
    server_tls_policy.name = 'name_value'
    request = network_security_v1beta1.UpdateServerTlsPolicyRequest(server_tls_policy=server_tls_policy)
    operation = client.update_server_tls_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)