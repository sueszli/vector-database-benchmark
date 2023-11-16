from google.cloud import beyondcorp_appconnections_v1

def sample_delete_app_connection():
    if False:
        return 10
    client = beyondcorp_appconnections_v1.AppConnectionsServiceClient()
    request = beyondcorp_appconnections_v1.DeleteAppConnectionRequest(name='name_value')
    operation = client.delete_app_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)