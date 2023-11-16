from google.cloud import dialogflowcx_v3

def sample_create_security_settings():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.SecuritySettingsServiceClient()
    security_settings = dialogflowcx_v3.SecuritySettings()
    security_settings.retention_window_days = 2271
    security_settings.display_name = 'display_name_value'
    request = dialogflowcx_v3.CreateSecuritySettingsRequest(parent='parent_value', security_settings=security_settings)
    response = client.create_security_settings(request=request)
    print(response)