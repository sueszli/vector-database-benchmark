from google.cloud import securitycenter_v1beta1

def sample_group_findings():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1beta1.SecurityCenterClient()
    request = securitycenter_v1beta1.GroupFindingsRequest(parent='parent_value', group_by='group_by_value')
    page_result = client.group_findings(request=request)
    for response in page_result:
        print(response)