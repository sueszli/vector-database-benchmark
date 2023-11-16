from google.cloud import dialogflowcx_v3beta1

def sample_get_security_settings():
    if False:
        return 10
    client = dialogflowcx_v3beta1.SecuritySettingsServiceClient()
    request = dialogflowcx_v3beta1.GetSecuritySettingsRequest(name='name_value')
    response = client.get_security_settings(request=request)
    print(response)