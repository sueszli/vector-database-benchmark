from google.cloud import dialogflowcx_v3beta1

def sample_update_security_settings():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.SecuritySettingsServiceClient()
    security_settings = dialogflowcx_v3beta1.SecuritySettings()
    security_settings.retention_window_days = 2271
    security_settings.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.UpdateSecuritySettingsRequest(security_settings=security_settings)
    response = client.update_security_settings(request=request)
    print(response)