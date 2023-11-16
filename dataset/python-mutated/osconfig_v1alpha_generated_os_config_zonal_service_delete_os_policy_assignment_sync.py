from google.cloud import osconfig_v1alpha

def sample_delete_os_policy_assignment():
    if False:
        for i in range(10):
            print('nop')
    client = osconfig_v1alpha.OsConfigZonalServiceClient()
    request = osconfig_v1alpha.DeleteOSPolicyAssignmentRequest(name='name_value')
    operation = client.delete_os_policy_assignment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)