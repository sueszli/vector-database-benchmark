from google.cloud import edgenetwork_v1

def sample_list_networks():
    if False:
        for i in range(10):
            print('nop')
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.ListNetworksRequest(parent='parent_value')
    page_result = client.list_networks(request=request)
    for response in page_result:
        print(response)