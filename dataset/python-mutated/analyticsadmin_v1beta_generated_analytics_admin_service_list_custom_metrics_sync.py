from google.analytics import admin_v1beta

def sample_list_custom_metrics():
    if False:
        i = 10
        return i + 15
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.ListCustomMetricsRequest(parent='parent_value')
    page_result = client.list_custom_metrics(request=request)
    for response in page_result:
        print(response)