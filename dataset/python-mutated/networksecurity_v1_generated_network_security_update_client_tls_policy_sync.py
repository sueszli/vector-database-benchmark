from google.cloud import network_security_v1

def sample_update_client_tls_policy():
    if False:
        while True:
            i = 10
    client = network_security_v1.NetworkSecurityClient()
    client_tls_policy = network_security_v1.ClientTlsPolicy()
    client_tls_policy.name = 'name_value'
    request = network_security_v1.UpdateClientTlsPolicyRequest(client_tls_policy=client_tls_policy)
    operation = client.update_client_tls_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)