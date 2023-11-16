from google.analytics import admin_v1beta

def sample_get_conversion_event():
    if False:
        while True:
            i = 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.GetConversionEventRequest(name='name_value')
    response = client.get_conversion_event(request=request)
    print(response)