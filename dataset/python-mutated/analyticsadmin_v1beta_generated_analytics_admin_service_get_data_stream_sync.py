from google.analytics import admin_v1beta

def sample_get_data_stream():
    if False:
        for i in range(10):
            print('nop')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.GetDataStreamRequest(name='name_value')
    response = client.get_data_stream(request=request)
    print(response)