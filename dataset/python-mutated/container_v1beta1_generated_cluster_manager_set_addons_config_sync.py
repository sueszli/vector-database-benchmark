from google.cloud import container_v1beta1

def sample_set_addons_config():
    if False:
        i = 10
        return i + 15
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.SetAddonsConfigRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value')
    response = client.set_addons_config(request=request)
    print(response)