from google.cloud import config_v1

def sample_get_resource():
    if False:
        i = 10
        return i + 15
    client = config_v1.ConfigClient()
    request = config_v1.GetResourceRequest(name='name_value')
    response = client.get_resource(request=request)
    print(response)