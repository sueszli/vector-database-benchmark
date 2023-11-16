from google.analytics import admin_v1beta

def sample_list_properties():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.ListPropertiesRequest(filter='filter_value')
    page_result = client.list_properties(request=request)
    for response in page_result:
        print(response)