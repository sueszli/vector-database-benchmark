from google.cloud import beyondcorp_appconnectors_v1

def sample_delete_app_connector():
    if False:
        while True:
            i = 10
    client = beyondcorp_appconnectors_v1.AppConnectorsServiceClient()
    request = beyondcorp_appconnectors_v1.DeleteAppConnectorRequest(name='name_value')
    operation = client.delete_app_connector(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)