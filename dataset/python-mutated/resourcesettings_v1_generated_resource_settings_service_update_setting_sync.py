from google.cloud import resourcesettings_v1

def sample_update_setting():
    if False:
        print('Hello World!')
    client = resourcesettings_v1.ResourceSettingsServiceClient()
    request = resourcesettings_v1.UpdateSettingRequest()
    response = client.update_setting(request=request)
    print(response)