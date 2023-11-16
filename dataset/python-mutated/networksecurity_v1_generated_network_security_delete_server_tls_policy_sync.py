from google.cloud import network_security_v1

def sample_delete_server_tls_policy():
    if False:
        while True:
            i = 10
    client = network_security_v1.NetworkSecurityClient()
    request = network_security_v1.DeleteServerTlsPolicyRequest(name='name_value')
    operation = client.delete_server_tls_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)