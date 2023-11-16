from google.cloud import clouddms_v1

def sample_update_conversion_workspace():
    if False:
        for i in range(10):
            print('nop')
    client = clouddms_v1.DataMigrationServiceClient()
    conversion_workspace = clouddms_v1.ConversionWorkspace()
    conversion_workspace.source.engine = 'ORACLE'
    conversion_workspace.source.version = 'version_value'
    conversion_workspace.destination.engine = 'ORACLE'
    conversion_workspace.destination.version = 'version_value'
    request = clouddms_v1.UpdateConversionWorkspaceRequest(conversion_workspace=conversion_workspace)
    operation = client.update_conversion_workspace(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)