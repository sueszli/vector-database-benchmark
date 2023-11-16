from google.cloud import osconfig_v1

def sample_list_os_policy_assignment_revisions():
    if False:
        for i in range(10):
            print('nop')
    client = osconfig_v1.OsConfigZonalServiceClient()
    request = osconfig_v1.ListOSPolicyAssignmentRevisionsRequest(name='name_value')
    page_result = client.list_os_policy_assignment_revisions(request=request)
    for response in page_result:
        print(response)