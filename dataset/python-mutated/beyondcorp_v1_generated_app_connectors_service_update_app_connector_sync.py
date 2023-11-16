from google.cloud import beyondcorp_appconnectors_v1

def sample_update_app_connector():
    if False:
        return 10
    client = beyondcorp_appconnectors_v1.AppConnectorsServiceClient()
    app_connector = beyondcorp_appconnectors_v1.AppConnector()
    app_connector.name = 'name_value'
    request = beyondcorp_appconnectors_v1.UpdateAppConnectorRequest(app_connector=app_connector)
    operation = client.update_app_connector(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)