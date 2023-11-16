from google.analytics import admin_v1beta

def sample_get_data_retention_settings():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.GetDataRetentionSettingsRequest(name='name_value')
    response = client.get_data_retention_settings(request=request)
    print(response)