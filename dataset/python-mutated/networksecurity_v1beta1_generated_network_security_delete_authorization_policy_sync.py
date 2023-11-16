from google.cloud import network_security_v1beta1

def sample_delete_authorization_policy():
    if False:
        i = 10
        return i + 15
    client = network_security_v1beta1.NetworkSecurityClient()
    request = network_security_v1beta1.DeleteAuthorizationPolicyRequest(name='name_value')
    operation = client.delete_authorization_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)