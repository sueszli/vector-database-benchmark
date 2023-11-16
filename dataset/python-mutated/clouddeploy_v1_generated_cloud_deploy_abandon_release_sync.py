from google.cloud import deploy_v1

def sample_abandon_release():
    if False:
        print('Hello World!')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.AbandonReleaseRequest(name='name_value')
    response = client.abandon_release(request=request)
    print(response)