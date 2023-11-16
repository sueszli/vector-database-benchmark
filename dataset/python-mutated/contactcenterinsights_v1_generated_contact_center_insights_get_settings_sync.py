from google.cloud import contact_center_insights_v1

def sample_get_settings():
    if False:
        while True:
            i = 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.GetSettingsRequest(name='name_value')
    response = client.get_settings(request=request)
    print(response)