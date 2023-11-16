from google.analytics import admin_v1beta

def sample_create_data_stream():
    if False:
        print('Hello World!')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    data_stream = admin_v1beta.DataStream()
    data_stream.type_ = 'IOS_APP_DATA_STREAM'
    request = admin_v1beta.CreateDataStreamRequest(parent='parent_value', data_stream=data_stream)
    response = client.create_data_stream(request=request)
    print(response)