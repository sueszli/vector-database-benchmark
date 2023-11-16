from google.analytics import admin_v1beta

def sample_delete_property():
    if False:
        while True:
            i = 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.DeletePropertyRequest(name='name_value')
    response = client.delete_property(request=request)
    print(response)