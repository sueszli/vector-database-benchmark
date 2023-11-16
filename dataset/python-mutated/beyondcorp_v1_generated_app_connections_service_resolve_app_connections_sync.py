from google.cloud import beyondcorp_appconnections_v1

def sample_resolve_app_connections():
    if False:
        print('Hello World!')
    client = beyondcorp_appconnections_v1.AppConnectionsServiceClient()
    request = beyondcorp_appconnections_v1.ResolveAppConnectionsRequest(parent='parent_value', app_connector_id='app_connector_id_value')
    page_result = client.resolve_app_connections(request=request)
    for response in page_result:
        print(response)