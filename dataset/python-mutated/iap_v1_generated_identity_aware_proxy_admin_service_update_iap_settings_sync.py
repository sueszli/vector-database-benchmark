from google.cloud import iap_v1

def sample_update_iap_settings():
    if False:
        return 10
    client = iap_v1.IdentityAwareProxyAdminServiceClient()
    iap_settings = iap_v1.IapSettings()
    iap_settings.name = 'name_value'
    request = iap_v1.UpdateIapSettingsRequest(iap_settings=iap_settings)
    response = client.update_iap_settings(request=request)
    print(response)