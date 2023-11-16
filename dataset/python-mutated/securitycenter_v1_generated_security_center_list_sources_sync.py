from google.cloud import securitycenter_v1

def sample_list_sources():
    if False:
        while True:
            i = 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.ListSourcesRequest(parent='parent_value')
    page_result = client.list_sources(request=request)
    for response in page_result:
        print(response)