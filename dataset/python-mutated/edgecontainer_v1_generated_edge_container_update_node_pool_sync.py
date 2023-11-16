from google.cloud import edgecontainer_v1

def sample_update_node_pool():
    if False:
        while True:
            i = 10
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.UpdateNodePoolRequest()
    operation = client.update_node_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)