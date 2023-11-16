from google.cloud import osconfig_v1alpha

def sample_list_os_policy_assignments():
    if False:
        while True:
            i = 10
    client = osconfig_v1alpha.OsConfigZonalServiceClient()
    request = osconfig_v1alpha.ListOSPolicyAssignmentsRequest(parent='parent_value')
    page_result = client.list_os_policy_assignments(request=request)
    for response in page_result:
        print(response)