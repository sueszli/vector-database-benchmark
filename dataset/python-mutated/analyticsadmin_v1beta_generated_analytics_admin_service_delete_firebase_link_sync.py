from google.analytics import admin_v1beta

def sample_delete_firebase_link():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.DeleteFirebaseLinkRequest(name='name_value')
    client.delete_firebase_link(request=request)