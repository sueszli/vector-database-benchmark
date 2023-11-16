from google.cloud import securitycenter_v1p1beta1

def sample_list_assets():
    if False:
        print('Hello World!')
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.ListAssetsRequest(parent='parent_value')
    page_result = client.list_assets(request=request)
    for response in page_result:
        print(response)