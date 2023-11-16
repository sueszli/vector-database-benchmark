from google.analytics import admin_v1beta

def sample_get_custom_dimension():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.GetCustomDimensionRequest(name='name_value')
    response = client.get_custom_dimension(request=request)
    print(response)