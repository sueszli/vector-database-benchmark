from google.analytics import admin_v1beta

def sample_create_conversion_event():
    if False:
        while True:
            i = 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.CreateConversionEventRequest(parent='parent_value')
    response = client.create_conversion_event(request=request)
    print(response)