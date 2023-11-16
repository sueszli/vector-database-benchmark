from google.cloud import iap_v1

def sample_list_identity_aware_proxy_clients():
    if False:
        print('Hello World!')
    client = iap_v1.IdentityAwareProxyOAuthServiceClient()
    request = iap_v1.ListIdentityAwareProxyClientsRequest(parent='parent_value')
    page_result = client.list_identity_aware_proxy_clients(request=request)
    for response in page_result:
        print(response)