from google.analytics import admin_v1beta

def sample_get_data_sharing_settings():
    if False:
        print('Hello World!')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.GetDataSharingSettingsRequest(name='name_value')
    response = client.get_data_sharing_settings(request=request)
    print(response)