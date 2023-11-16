from google.cloud import osconfig_v1

def sample_delete_os_policy_assignment():
    if False:
        i = 10
        return i + 15
    client = osconfig_v1.OsConfigZonalServiceClient()
    request = osconfig_v1.DeleteOSPolicyAssignmentRequest(name='name_value')
    operation = client.delete_os_policy_assignment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)