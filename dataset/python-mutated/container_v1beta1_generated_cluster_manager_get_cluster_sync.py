from google.cloud import container_v1beta1

def sample_get_cluster():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.GetClusterRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value')
    response = client.get_cluster(request=request)
    print(response)