from google.cloud import osconfig_v1

def sample_get_os_policy_assignment():
    if False:
        print('Hello World!')
    client = osconfig_v1.OsConfigZonalServiceClient()
    request = osconfig_v1.GetOSPolicyAssignmentRequest(name='name_value')
    response = client.get_os_policy_assignment(request=request)
    print(response)