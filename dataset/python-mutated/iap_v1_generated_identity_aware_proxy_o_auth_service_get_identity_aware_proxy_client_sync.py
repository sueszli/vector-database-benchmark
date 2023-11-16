from google.cloud import iap_v1

def sample_get_identity_aware_proxy_client():
    if False:
        for i in range(10):
            print('nop')
    client = iap_v1.IdentityAwareProxyOAuthServiceClient()
    request = iap_v1.GetIdentityAwareProxyClientRequest(name='name_value')
    response = client.get_identity_aware_proxy_client(request=request)
    print(response)