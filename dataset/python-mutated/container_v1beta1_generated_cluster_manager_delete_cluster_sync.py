from google.cloud import container_v1beta1

def sample_delete_cluster():
    if False:
        i = 10
        return i + 15
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.DeleteClusterRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value')
    response = client.delete_cluster(request=request)
    print(response)