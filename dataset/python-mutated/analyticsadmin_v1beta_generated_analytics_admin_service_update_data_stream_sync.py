from google.analytics import admin_v1beta

def sample_update_data_stream():
    if False:
        i = 10
        return i + 15
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.UpdateDataStreamRequest()
    response = client.update_data_stream(request=request)
    print(response)