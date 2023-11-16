from google.analytics import admin_v1beta

def sample_list_firebase_links():
    if False:
        while True:
            i = 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.ListFirebaseLinksRequest(parent='parent_value')
    page_result = client.list_firebase_links(request=request)
    for response in page_result:
        print(response)