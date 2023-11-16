from google.cloud import clouddms_v1

def sample_commit_conversion_workspace():
    if False:
        i = 10
        return i + 15
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.CommitConversionWorkspaceRequest(name='name_value')
    operation = client.commit_conversion_workspace(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)