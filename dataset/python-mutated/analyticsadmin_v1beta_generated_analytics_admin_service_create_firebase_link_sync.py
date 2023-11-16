from google.analytics import admin_v1beta

def sample_create_firebase_link():
    if False:
        for i in range(10):
            print('nop')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.CreateFirebaseLinkRequest(parent='parent_value')
    response = client.create_firebase_link(request=request)
    print(response)