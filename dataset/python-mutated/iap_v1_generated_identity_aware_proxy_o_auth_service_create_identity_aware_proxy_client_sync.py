from google.cloud import iap_v1

def sample_create_identity_aware_proxy_client():
    if False:
        while True:
            i = 10
    client = iap_v1.IdentityAwareProxyOAuthServiceClient()
    request = iap_v1.CreateIdentityAwareProxyClientRequest(parent='parent_value')
    response = client.create_identity_aware_proxy_client(request=request)
    print(response)