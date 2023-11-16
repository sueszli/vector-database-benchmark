from google.analytics import admin_v1beta

def sample_get_property():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.GetPropertyRequest(name='name_value')
    response = client.get_property(request=request)
    print(response)