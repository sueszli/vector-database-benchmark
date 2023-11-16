from google.cloud import dialogflowcx_v3beta1

def sample_delete_security_settings():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.SecuritySettingsServiceClient()
    request = dialogflowcx_v3beta1.DeleteSecuritySettingsRequest(name='name_value')
    client.delete_security_settings(request=request)