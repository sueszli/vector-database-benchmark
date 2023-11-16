from google.analytics import admin_v1beta

def sample_update_custom_dimension():
    if False:
        while True:
            i = 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.UpdateCustomDimensionRequest()
    response = client.update_custom_dimension(request=request)
    print(response)