from google.analytics import admin_v1beta

def sample_update_custom_metric():
    if False:
        print('Hello World!')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.UpdateCustomMetricRequest()
    response = client.update_custom_metric(request=request)
    print(response)