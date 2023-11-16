from google.analytics import admin_v1beta

def sample_delete_data_stream():
    if False:
        while True:
            i = 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.DeleteDataStreamRequest(name='name_value')
    client.delete_data_stream(request=request)