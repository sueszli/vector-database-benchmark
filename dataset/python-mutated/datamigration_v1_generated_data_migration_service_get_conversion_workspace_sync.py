from google.cloud import clouddms_v1

def sample_get_conversion_workspace():
    if False:
        for i in range(10):
            print('nop')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.GetConversionWorkspaceRequest(name='name_value')
    response = client.get_conversion_workspace(request=request)
    print(response)