from google.cloud import container_v1

def sample_get_json_web_keys():
    if False:
        i = 10
        return i + 15
    client = container_v1.ClusterManagerClient()
    request = container_v1.GetJSONWebKeysRequest()
    response = client.get_json_web_keys(request=request)
    print(response)