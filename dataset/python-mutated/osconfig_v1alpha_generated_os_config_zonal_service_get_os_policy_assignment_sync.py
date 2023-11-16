from google.cloud import osconfig_v1alpha

def sample_get_os_policy_assignment():
    if False:
        print('Hello World!')
    client = osconfig_v1alpha.OsConfigZonalServiceClient()
    request = osconfig_v1alpha.GetOSPolicyAssignmentRequest(name='name_value')
    response = client.get_os_policy_assignment(request=request)
    print(response)