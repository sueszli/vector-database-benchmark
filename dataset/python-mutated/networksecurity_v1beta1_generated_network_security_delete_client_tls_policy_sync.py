from google.cloud import network_security_v1beta1

def sample_delete_client_tls_policy():
    if False:
        return 10
    client = network_security_v1beta1.NetworkSecurityClient()
    request = network_security_v1beta1.DeleteClientTlsPolicyRequest(name='name_value')
    operation = client.delete_client_tls_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)