from google.cloud import edgecontainer_v1

def sample_get_node_pool():
    if False:
        print('Hello World!')
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.GetNodePoolRequest(name='name_value')
    response = client.get_node_pool(request=request)
    print(response)