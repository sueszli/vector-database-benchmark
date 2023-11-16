from google.cloud import iap_v1

def sample_reset_identity_aware_proxy_client_secret():
    if False:
        print('Hello World!')
    client = iap_v1.IdentityAwareProxyOAuthServiceClient()
    request = iap_v1.ResetIdentityAwareProxyClientSecretRequest(name='name_value')
    response = client.reset_identity_aware_proxy_client_secret(request=request)
    print(response)