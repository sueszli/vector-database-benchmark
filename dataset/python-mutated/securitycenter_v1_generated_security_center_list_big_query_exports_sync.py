from google.cloud import securitycenter_v1

def sample_list_big_query_exports():
    if False:
        print('Hello World!')
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.ListBigQueryExportsRequest(parent='parent_value')
    page_result = client.list_big_query_exports(request=request)
    for response in page_result:
        print(response)