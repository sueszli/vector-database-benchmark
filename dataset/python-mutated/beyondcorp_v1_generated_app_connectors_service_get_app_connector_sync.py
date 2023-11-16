from google.cloud import beyondcorp_appconnectors_v1

def sample_get_app_connector():
    if False:
        print('Hello World!')
    client = beyondcorp_appconnectors_v1.AppConnectorsServiceClient()
    request = beyondcorp_appconnectors_v1.GetAppConnectorRequest(name='name_value')
    response = client.get_app_connector(request=request)
    print(response)