from google.cloud import securitycenter_v1

def sample_group_findings():
    if False:
        print('Hello World!')
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.GroupFindingsRequest(parent='parent_value', group_by='group_by_value')
    page_result = client.group_findings(request=request)
    for response in page_result:
        print(response)