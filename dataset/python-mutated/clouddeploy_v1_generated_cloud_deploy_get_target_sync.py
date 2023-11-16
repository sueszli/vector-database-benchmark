from google.cloud import deploy_v1

def sample_get_target():
    if False:
        while True:
            i = 10
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.GetTargetRequest(name='name_value')
    response = client.get_target(request=request)
    print(response)