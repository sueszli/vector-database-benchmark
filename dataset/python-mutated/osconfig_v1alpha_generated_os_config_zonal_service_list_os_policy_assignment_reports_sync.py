from google.cloud import osconfig_v1alpha

def sample_list_os_policy_assignment_reports():
    if False:
        print('Hello World!')
    client = osconfig_v1alpha.OsConfigZonalServiceClient()
    request = osconfig_v1alpha.ListOSPolicyAssignmentReportsRequest(parent='parent_value')
    page_result = client.list_os_policy_assignment_reports(request=request)
    for response in page_result:
        print(response)