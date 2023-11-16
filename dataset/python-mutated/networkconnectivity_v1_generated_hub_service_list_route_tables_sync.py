from google.cloud import networkconnectivity_v1

def sample_list_route_tables():
    if False:
        i = 10
        return i + 15
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.ListRouteTablesRequest(parent='parent_value')
    page_result = client.list_route_tables(request=request)
    for response in page_result:
        print(response)