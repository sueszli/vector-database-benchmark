from google.cloud import dialogflowcx_v3beta1

def sample_list_security_settings():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.SecuritySettingsServiceClient()
    request = dialogflowcx_v3beta1.ListSecuritySettingsRequest(parent='parent_value')
    page_result = client.list_security_settings(request=request)
    for response in page_result:
        print(response)