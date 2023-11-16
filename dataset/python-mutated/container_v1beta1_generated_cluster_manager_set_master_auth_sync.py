from google.cloud import container_v1beta1

def sample_set_master_auth():
    if False:
        print('Hello World!')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.SetMasterAuthRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', action='SET_USERNAME')
    response = client.set_master_auth(request=request)
    print(response)