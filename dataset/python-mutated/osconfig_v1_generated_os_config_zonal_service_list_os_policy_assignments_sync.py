from google.cloud import osconfig_v1

def sample_list_os_policy_assignments():
    if False:
        return 10
    client = osconfig_v1.OsConfigZonalServiceClient()
    request = osconfig_v1.ListOSPolicyAssignmentsRequest(parent='parent_value')
    page_result = client.list_os_policy_assignments(request=request)
    for response in page_result:
        print(response)