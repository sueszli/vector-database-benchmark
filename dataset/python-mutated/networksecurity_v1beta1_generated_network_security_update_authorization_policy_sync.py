from google.cloud import network_security_v1beta1

def sample_update_authorization_policy():
    if False:
        return 10
    client = network_security_v1beta1.NetworkSecurityClient()
    authorization_policy = network_security_v1beta1.AuthorizationPolicy()
    authorization_policy.name = 'name_value'
    authorization_policy.action = 'DENY'
    request = network_security_v1beta1.UpdateAuthorizationPolicyRequest(authorization_policy=authorization_policy)
    operation = client.update_authorization_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)