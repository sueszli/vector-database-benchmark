from google.cloud import securitycenter_v1beta1

def sample_list_findings():
    if False:
        print('Hello World!')
    client = securitycenter_v1beta1.SecurityCenterClient()
    request = securitycenter_v1beta1.ListFindingsRequest(parent='parent_value')
    page_result = client.list_findings(request=request)
    for response in page_result:
        print(response)