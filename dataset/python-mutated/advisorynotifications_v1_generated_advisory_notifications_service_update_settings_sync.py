from google.cloud import advisorynotifications_v1

def sample_update_settings():
    if False:
        while True:
            i = 10
    client = advisorynotifications_v1.AdvisoryNotificationsServiceClient()
    settings = advisorynotifications_v1.Settings()
    settings.etag = 'etag_value'
    request = advisorynotifications_v1.UpdateSettingsRequest(settings=settings)
    response = client.update_settings(request=request)
    print(response)