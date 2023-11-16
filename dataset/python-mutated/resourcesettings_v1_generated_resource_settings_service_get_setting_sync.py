from google.cloud import resourcesettings_v1

def sample_get_setting():
    if False:
        i = 10
        return i + 15
    client = resourcesettings_v1.ResourceSettingsServiceClient()
    request = resourcesettings_v1.GetSettingRequest(name='name_value')
    response = client.get_setting(request=request)
    print(response)