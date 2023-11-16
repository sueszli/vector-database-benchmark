from google.cloud import beyondcorp_appconnections_v1

def sample_get_app_connection():
    if False:
        i = 10
        return i + 15
    client = beyondcorp_appconnections_v1.AppConnectionsServiceClient()
    request = beyondcorp_appconnections_v1.GetAppConnectionRequest(name='name_value')
    response = client.get_app_connection(request=request)
    print(response)