from google.cloud import network_security_v1

def sample_create_authorization_policy():
    if False:
        print('Hello World!')
    client = network_security_v1.NetworkSecurityClient()
    authorization_policy = network_security_v1.AuthorizationPolicy()
    authorization_policy.name = 'name_value'
    authorization_policy.action = 'DENY'
    request = network_security_v1.CreateAuthorizationPolicyRequest(parent='parent_value', authorization_policy_id='authorization_policy_id_value', authorization_policy=authorization_policy)
    operation = client.create_authorization_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)