from google.cloud import clouddms_v1

def sample_delete_private_connection():
    if False:
        return 10
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.DeletePrivateConnectionRequest(name='name_value')
    operation = client.delete_private_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)