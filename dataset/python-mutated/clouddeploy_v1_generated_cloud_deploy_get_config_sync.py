from google.cloud import deploy_v1

def sample_get_config():
    if False:
        return 10
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.GetConfigRequest(name='name_value')
    response = client.get_config(request=request)
    print(response)