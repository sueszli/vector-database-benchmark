from google.analytics import admin_v1beta

def sample_archive_custom_metric():
    if False:
        print('Hello World!')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.ArchiveCustomMetricRequest(name='name_value')
    client.archive_custom_metric(request=request)