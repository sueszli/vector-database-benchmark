from google.cloud import beyondcorp_appconnections_v1

def sample_list_app_connections():
    if False:
        while True:
            i = 10
    client = beyondcorp_appconnections_v1.AppConnectionsServiceClient()
    request = beyondcorp_appconnections_v1.ListAppConnectionsRequest(parent='parent_value')
    page_result = client.list_app_connections(request=request)
    for response in page_result:
        print(response)