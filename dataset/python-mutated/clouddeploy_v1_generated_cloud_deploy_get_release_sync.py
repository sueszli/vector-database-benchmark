from google.cloud import deploy_v1

def sample_get_release():
    if False:
        for i in range(10):
            print('nop')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.GetReleaseRequest(name='name_value')
    response = client.get_release(request=request)
    print(response)