from google.cloud import container_v1

def sample_set_addons_config():
    if False:
        i = 10
        return i + 15
    client = container_v1.ClusterManagerClient()
    request = container_v1.SetAddonsConfigRequest()
    response = client.set_addons_config(request=request)
    print(response)