from google.cloud import iap_v1

def sample_get_iap_settings():
    if False:
        for i in range(10):
            print('nop')
    client = iap_v1.IdentityAwareProxyAdminServiceClient()
    request = iap_v1.GetIapSettingsRequest(name='name_value')
    response = client.get_iap_settings(request=request)
    print(response)