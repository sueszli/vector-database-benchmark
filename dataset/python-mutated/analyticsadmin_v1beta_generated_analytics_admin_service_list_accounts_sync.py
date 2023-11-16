from google.analytics import admin_v1beta

def sample_list_accounts():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.ListAccountsRequest()
    page_result = client.list_accounts(request=request)
    for response in page_result:
        print(response)