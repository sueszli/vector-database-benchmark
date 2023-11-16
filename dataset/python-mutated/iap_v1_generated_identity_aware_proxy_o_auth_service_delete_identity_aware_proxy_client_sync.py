from google.cloud import iap_v1

def sample_delete_identity_aware_proxy_client():
    if False:
        print('Hello World!')
    client = iap_v1.IdentityAwareProxyOAuthServiceClient()
    request = iap_v1.DeleteIdentityAwareProxyClientRequest(name='name_value')
    client.delete_identity_aware_proxy_client(request=request)