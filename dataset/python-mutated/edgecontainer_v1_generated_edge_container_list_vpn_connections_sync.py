from google.cloud import edgecontainer_v1

def sample_list_vpn_connections():
    if False:
        for i in range(10):
            print('nop')
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.ListVpnConnectionsRequest(parent='parent_value')
    page_result = client.list_vpn_connections(request=request)
    for response in page_result:
        print(response)