from google.cloud import edgecontainer_v1

def sample_create_cluster():
    if False:
        i = 10
        return i + 15
    client = edgecontainer_v1.EdgeContainerClient()
    cluster = edgecontainer_v1.Cluster()
    cluster.name = 'name_value'
    cluster.networking.cluster_ipv4_cidr_blocks = ['cluster_ipv4_cidr_blocks_value1', 'cluster_ipv4_cidr_blocks_value2']
    cluster.networking.services_ipv4_cidr_blocks = ['services_ipv4_cidr_blocks_value1', 'services_ipv4_cidr_blocks_value2']
    cluster.authorization.admin_users.username = 'username_value'
    request = edgecontainer_v1.CreateClusterRequest(parent='parent_value', cluster_id='cluster_id_value', cluster=cluster)
    operation = client.create_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)