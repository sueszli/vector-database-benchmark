from google.cloud import securitycenter_v1p1beta1

def sample_list_sources():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.ListSourcesRequest(parent='parent_value')
    page_result = client.list_sources(request=request)
    for response in page_result:
        print(response)