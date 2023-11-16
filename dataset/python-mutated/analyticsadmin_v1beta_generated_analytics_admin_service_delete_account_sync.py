from google.analytics import admin_v1beta

def sample_delete_account():
    if False:
        i = 10
        return i + 15
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.DeleteAccountRequest(name='name_value')
    client.delete_account(request=request)