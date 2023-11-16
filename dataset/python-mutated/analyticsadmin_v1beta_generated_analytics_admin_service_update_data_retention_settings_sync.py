from google.analytics import admin_v1beta

def sample_update_data_retention_settings():
    if False:
        print('Hello World!')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.UpdateDataRetentionSettingsRequest()
    response = client.update_data_retention_settings(request=request)
    print(response)