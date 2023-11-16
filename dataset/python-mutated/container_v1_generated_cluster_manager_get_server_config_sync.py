from google.cloud import container_v1

def sample_get_server_config():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1.ClusterManagerClient()
    request = container_v1.GetServerConfigRequest()
    response = client.get_server_config(request=request)
    print(response)