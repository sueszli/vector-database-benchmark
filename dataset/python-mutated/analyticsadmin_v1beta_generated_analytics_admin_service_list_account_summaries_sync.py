from google.analytics import admin_v1beta

def sample_list_account_summaries():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.ListAccountSummariesRequest()
    page_result = client.list_account_summaries(request=request)
    for response in page_result:
        print(response)