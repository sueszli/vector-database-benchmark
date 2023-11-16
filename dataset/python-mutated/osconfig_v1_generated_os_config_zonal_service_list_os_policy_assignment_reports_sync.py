from google.cloud import osconfig_v1

def sample_list_os_policy_assignment_reports():
    if False:
        for i in range(10):
            print('nop')
    client = osconfig_v1.OsConfigZonalServiceClient()
    request = osconfig_v1.ListOSPolicyAssignmentReportsRequest(parent='parent_value')
    page_result = client.list_os_policy_assignment_reports(request=request)
    for response in page_result:
        print(response)