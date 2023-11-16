from google.cloud import network_security_v1beta1

def sample_update_client_tls_policy():
    if False:
        print('Hello World!')
    client = network_security_v1beta1.NetworkSecurityClient()
    client_tls_policy = network_security_v1beta1.ClientTlsPolicy()
    client_tls_policy.name = 'name_value'
    request = network_security_v1beta1.UpdateClientTlsPolicyRequest(client_tls_policy=client_tls_policy)
    operation = client.update_client_tls_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)