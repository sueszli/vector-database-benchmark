from google.cloud import resourcesettings_v1

def sample_list_settings():
    if False:
        print('Hello World!')
    client = resourcesettings_v1.ResourceSettingsServiceClient()
    request = resourcesettings_v1.ListSettingsRequest(parent='parent_value')
    page_result = client.list_settings(request=request)
    for response in page_result:
        print(response)