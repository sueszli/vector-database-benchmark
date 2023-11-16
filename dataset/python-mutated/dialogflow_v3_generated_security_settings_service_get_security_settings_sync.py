from google.cloud import dialogflowcx_v3

def sample_get_security_settings():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.SecuritySettingsServiceClient()
    request = dialogflowcx_v3.GetSecuritySettingsRequest(name='name_value')
    response = client.get_security_settings(request=request)
    print(response)