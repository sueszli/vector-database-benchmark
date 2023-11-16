from google.cloud import networkconnectivity_v1

def sample_list_routes():
    if False:
        while True:
            i = 10
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.ListRoutesRequest(parent='parent_value')
    page_result = client.list_routes(request=request)
    for response in page_result:
        print(response)