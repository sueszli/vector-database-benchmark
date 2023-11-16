from google.cloud import network_security_v1beta1

def sample_list_client_tls_policies():
    if False:
        while True:
            i = 10
    client = network_security_v1beta1.NetworkSecurityClient()
    request = network_security_v1beta1.ListClientTlsPoliciesRequest(parent='parent_value')
    page_result = client.list_client_tls_policies(request=request)
    for response in page_result:
        print(response)