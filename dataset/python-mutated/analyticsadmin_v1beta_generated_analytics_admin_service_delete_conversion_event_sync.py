from google.analytics import admin_v1beta

def sample_delete_conversion_event():
    if False:
        for i in range(10):
            print('nop')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.DeleteConversionEventRequest(name='name_value')
    client.delete_conversion_event(request=request)