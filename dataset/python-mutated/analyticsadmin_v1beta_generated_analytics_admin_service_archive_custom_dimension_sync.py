from google.analytics import admin_v1beta

def sample_archive_custom_dimension():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.ArchiveCustomDimensionRequest(name='name_value')
    client.archive_custom_dimension(request=request)