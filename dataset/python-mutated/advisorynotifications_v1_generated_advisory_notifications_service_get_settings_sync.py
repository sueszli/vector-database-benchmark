from google.cloud import advisorynotifications_v1

def sample_get_settings():
    if False:
        while True:
            i = 10
    client = advisorynotifications_v1.AdvisoryNotificationsServiceClient()
    request = advisorynotifications_v1.GetSettingsRequest(name='name_value')
    response = client.get_settings(request=request)
    print(response)