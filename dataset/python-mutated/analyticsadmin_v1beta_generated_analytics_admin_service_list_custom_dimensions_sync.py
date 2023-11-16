from google.analytics import admin_v1beta

def sample_list_custom_dimensions():
    if False:
        for i in range(10):
            print('nop')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.ListCustomDimensionsRequest(parent='parent_value')
    page_result = client.list_custom_dimensions(request=request)
    for response in page_result:
        print(response)