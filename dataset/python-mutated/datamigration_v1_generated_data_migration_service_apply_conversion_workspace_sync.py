from google.cloud import clouddms_v1

def sample_apply_conversion_workspace():
    if False:
        i = 10
        return i + 15
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.ApplyConversionWorkspaceRequest(connection_profile='connection_profile_value', name='name_value')
    operation = client.apply_conversion_workspace(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)