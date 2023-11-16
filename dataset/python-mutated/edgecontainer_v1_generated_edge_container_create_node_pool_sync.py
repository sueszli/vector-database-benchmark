from google.cloud import edgecontainer_v1

def sample_create_node_pool():
    if False:
        while True:
            i = 10
    client = edgecontainer_v1.EdgeContainerClient()
    node_pool = edgecontainer_v1.NodePool()
    node_pool.name = 'name_value'
    node_pool.node_count = 1070
    request = edgecontainer_v1.CreateNodePoolRequest(parent='parent_value', node_pool_id='node_pool_id_value', node_pool=node_pool)
    operation = client.create_node_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)