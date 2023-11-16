from google.cloud import network_security_v1

def sample_list_authorization_policies():
    if False:
        return 10
    client = network_security_v1.NetworkSecurityClient()
    request = network_security_v1.ListAuthorizationPoliciesRequest(parent='parent_value')
    page_result = client.list_authorization_policies(request=request)
    for response in page_result:
        print(response)