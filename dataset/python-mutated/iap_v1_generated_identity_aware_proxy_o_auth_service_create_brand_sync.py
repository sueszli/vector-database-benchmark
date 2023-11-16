from google.cloud import iap_v1

def sample_create_brand():
    if False:
        while True:
            i = 10
    client = iap_v1.IdentityAwareProxyOAuthServiceClient()
    request = iap_v1.CreateBrandRequest(parent='parent_value')
    response = client.create_brand(request=request)
    print(response)