from google.analytics import admin_v1beta

def sample_update_conversion_event():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.UpdateConversionEventRequest()
    response = client.update_conversion_event(request=request)
    print(response)