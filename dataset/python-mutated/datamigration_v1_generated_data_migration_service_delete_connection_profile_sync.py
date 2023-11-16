from google.cloud import clouddms_v1

def sample_delete_connection_profile():
    if False:
        for i in range(10):
            print('nop')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.DeleteConnectionProfileRequest(name='name_value')
    operation = client.delete_connection_profile(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)