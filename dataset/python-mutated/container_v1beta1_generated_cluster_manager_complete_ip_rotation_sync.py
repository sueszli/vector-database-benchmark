from google.cloud import container_v1beta1

def sample_complete_ip_rotation():
    if False:
        while True:
            i = 10
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.CompleteIPRotationRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value')
    response = client.complete_ip_rotation(request=request)
    print(response)