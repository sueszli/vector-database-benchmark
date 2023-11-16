from google.cloud import container_v1beta1

def sample_start_ip_rotation():
    if False:
        i = 10
        return i + 15
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.StartIPRotationRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value')
    response = client.start_ip_rotation(request=request)
    print(response)