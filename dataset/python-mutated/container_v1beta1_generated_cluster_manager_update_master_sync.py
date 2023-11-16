from google.cloud import container_v1beta1

def sample_update_master():
    if False:
        i = 10
        return i + 15
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.UpdateMasterRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', master_version='master_version_value')
    response = client.update_master(request=request)
    print(response)