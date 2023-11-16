from google.cloud import config_v1

def sample_get_deployment():
    if False:
        print('Hello World!')
    client = config_v1.ConfigClient()
    request = config_v1.GetDeploymentRequest(name='name_value')
    response = client.get_deployment(request=request)
    print(response)