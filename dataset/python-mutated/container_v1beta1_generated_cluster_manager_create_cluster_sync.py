from google.cloud import container_v1beta1

def sample_create_cluster():
    if False:
        i = 10
        return i + 15
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.CreateClusterRequest(project_id='project_id_value', zone='zone_value')
    response = client.create_cluster(request=request)
    print(response)