from google.cloud import container_v1beta1

def sample_create_node_pool():
    if False:
        print('Hello World!')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.CreateNodePoolRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value')
    response = client.create_node_pool(request=request)
    print(response)