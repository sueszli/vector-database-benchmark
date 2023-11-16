from google.cloud import network_security_v1

def sample_create_server_tls_policy():
    if False:
        return 10
    client = network_security_v1.NetworkSecurityClient()
    server_tls_policy = network_security_v1.ServerTlsPolicy()
    server_tls_policy.name = 'name_value'
    request = network_security_v1.CreateServerTlsPolicyRequest(parent='parent_value', server_tls_policy_id='server_tls_policy_id_value', server_tls_policy=server_tls_policy)
    operation = client.create_server_tls_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)