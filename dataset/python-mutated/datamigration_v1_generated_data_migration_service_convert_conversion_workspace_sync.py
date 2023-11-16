from google.cloud import clouddms_v1

def sample_convert_conversion_workspace():
    if False:
        while True:
            i = 10
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.ConvertConversionWorkspaceRequest()
    operation = client.convert_conversion_workspace(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)