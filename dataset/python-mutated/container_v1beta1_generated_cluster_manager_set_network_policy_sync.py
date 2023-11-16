from google.cloud import container_v1beta1

def sample_set_network_policy():
    if False:
        i = 10
        return i + 15
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.SetNetworkPolicyRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value')
    response = client.set_network_policy(request=request)
    print(response)