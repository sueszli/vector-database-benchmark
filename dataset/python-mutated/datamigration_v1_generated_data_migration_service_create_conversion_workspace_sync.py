from google.cloud import clouddms_v1

def sample_create_conversion_workspace():
    if False:
        print('Hello World!')
    client = clouddms_v1.DataMigrationServiceClient()
    conversion_workspace = clouddms_v1.ConversionWorkspace()
    conversion_workspace.source.engine = 'ORACLE'
    conversion_workspace.source.version = 'version_value'
    conversion_workspace.destination.engine = 'ORACLE'
    conversion_workspace.destination.version = 'version_value'
    request = clouddms_v1.CreateConversionWorkspaceRequest(parent='parent_value', conversion_workspace_id='conversion_workspace_id_value', conversion_workspace=conversion_workspace)
    operation = client.create_conversion_workspace(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)