from google.cloud import iap_v1

def sample_list_brands():
    if False:
        i = 10
        return i + 15
    client = iap_v1.IdentityAwareProxyOAuthServiceClient()
    request = iap_v1.ListBrandsRequest(parent='parent_value')
    response = client.list_brands(request=request)
    print(response)