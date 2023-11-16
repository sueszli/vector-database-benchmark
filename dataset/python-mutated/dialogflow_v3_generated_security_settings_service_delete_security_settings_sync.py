from google.cloud import dialogflowcx_v3

def sample_delete_security_settings():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.SecuritySettingsServiceClient()
    request = dialogflowcx_v3.DeleteSecuritySettingsRequest(name='name_value')
    client.delete_security_settings(request=request)