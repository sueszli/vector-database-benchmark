from google.cloud import deploy_v1

def sample_get_rollout():
    if False:
        return 10
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.GetRolloutRequest(name='name_value')
    response = client.get_rollout(request=request)
    print(response)