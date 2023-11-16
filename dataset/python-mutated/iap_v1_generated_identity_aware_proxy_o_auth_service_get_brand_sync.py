from google.cloud import iap_v1

def sample_get_brand():
    if False:
        while True:
            i = 10
    client = iap_v1.IdentityAwareProxyOAuthServiceClient()
    request = iap_v1.GetBrandRequest(name='name_value')
    response = client.get_brand(request=request)
    print(response)