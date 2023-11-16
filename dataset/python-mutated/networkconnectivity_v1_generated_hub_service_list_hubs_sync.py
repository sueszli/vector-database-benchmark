from google.cloud import networkconnectivity_v1

def sample_list_hubs():
    if False:
        for i in range(10):
            print('nop')
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.ListHubsRequest(parent='parent_value')
    page_result = client.list_hubs(request=request)
    for response in page_result:
        print(response)