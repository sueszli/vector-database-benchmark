from google.cloud import edgecontainer_v1

def sample_delete_node_pool():
    if False:
        return 10
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.DeleteNodePoolRequest(name='name_value')
    operation = client.delete_node_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)