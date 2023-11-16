from google.cloud import container_v1beta1

def sample_get_server_config():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.GetServerConfigRequest(project_id='project_id_value', zone='zone_value')
    response = client.get_server_config(request=request)
    print(response)