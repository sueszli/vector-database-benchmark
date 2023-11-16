from google.analytics import admin_v1beta

def sample_list_conversion_events():
    if False:
        print('Hello World!')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.ListConversionEventsRequest(parent='parent_value')
    page_result = client.list_conversion_events(request=request)
    for response in page_result:
        print(response)