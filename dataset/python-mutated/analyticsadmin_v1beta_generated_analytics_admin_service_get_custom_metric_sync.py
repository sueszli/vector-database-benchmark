from google.analytics import admin_v1beta

def sample_get_custom_metric():
    if False:
        i = 10
        return i + 15
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.GetCustomMetricRequest(name='name_value')
    response = client.get_custom_metric(request=request)
    print(response)