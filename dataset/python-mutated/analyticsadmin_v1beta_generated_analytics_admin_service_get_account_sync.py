from google.analytics import admin_v1beta

def sample_get_account():
    if False:
        print('Hello World!')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.GetAccountRequest(name='name_value')
    response = client.get_account(request=request)
    print(response)