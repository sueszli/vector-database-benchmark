from google.analytics import admin_v1beta

def sample_update_account():
    if False:
        i = 10
        return i + 15
    client = admin_v1beta.AnalyticsAdminServiceClient()
    account = admin_v1beta.Account()
    account.display_name = 'display_name_value'
    request = admin_v1beta.UpdateAccountRequest(account=account)
    response = client.update_account(request=request)
    print(response)