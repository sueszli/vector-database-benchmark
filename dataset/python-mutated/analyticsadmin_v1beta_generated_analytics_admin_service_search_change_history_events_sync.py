from google.analytics import admin_v1beta

def sample_search_change_history_events():
    if False:
        print('Hello World!')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.SearchChangeHistoryEventsRequest(account='account_value')
    page_result = client.search_change_history_events(request=request)
    for response in page_result:
        print(response)