from google.cloud import networkconnectivity_v1

def sample_list_groups():
    if False:
        while True:
            i = 10
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.ListGroupsRequest(parent='parent_value')
    page_result = client.list_groups(request=request)
    for response in page_result:
        print(response)