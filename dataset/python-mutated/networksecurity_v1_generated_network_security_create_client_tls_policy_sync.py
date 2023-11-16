from google.cloud import network_security_v1

def sample_create_client_tls_policy():
    if False:
        for i in range(10):
            print('nop')
    client = network_security_v1.NetworkSecurityClient()
    client_tls_policy = network_security_v1.ClientTlsPolicy()
    client_tls_policy.name = 'name_value'
    request = network_security_v1.CreateClientTlsPolicyRequest(parent='parent_value', client_tls_policy_id='client_tls_policy_id_value', client_tls_policy=client_tls_policy)
    operation = client.create_client_tls_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)