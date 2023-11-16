from google.cloud import edgecontainer_v1

def sample_list_node_pools():
    if False:
        for i in range(10):
            print('nop')
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.ListNodePoolsRequest(parent='parent_value')
    page_result = client.list_node_pools(request=request)
    for response in page_result:
        print(response)