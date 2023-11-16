from google.cloud import securitycenter_v1

def sample_group_assets():
    if False:
        print('Hello World!')
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.GroupAssetsRequest(parent='parent_value', group_by='group_by_value')
    page_result = client.group_assets(request=request)
    for response in page_result:
        print(response)