from google.cloud import dialogflowcx_v3

def sample_list_security_settings():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.SecuritySettingsServiceClient()
    request = dialogflowcx_v3.ListSecuritySettingsRequest(parent='parent_value')
    page_result = client.list_security_settings(request=request)
    for response in page_result:
        print(response)