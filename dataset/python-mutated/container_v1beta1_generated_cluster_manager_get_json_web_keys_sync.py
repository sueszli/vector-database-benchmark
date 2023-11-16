from google.cloud import container_v1beta1

def sample_get_json_web_keys():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.GetJSONWebKeysRequest()
    response = client.get_json_web_keys(request=request)
    print(response)