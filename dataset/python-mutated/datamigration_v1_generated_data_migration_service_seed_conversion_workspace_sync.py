from google.cloud import clouddms_v1

def sample_seed_conversion_workspace():
    if False:
        for i in range(10):
            print('nop')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.SeedConversionWorkspaceRequest(source_connection_profile='source_connection_profile_value')
    operation = client.seed_conversion_workspace(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)