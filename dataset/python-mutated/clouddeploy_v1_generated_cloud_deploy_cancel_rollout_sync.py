from google.cloud import deploy_v1

def sample_cancel_rollout():
    if False:
        while True:
            i = 10
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.CancelRolloutRequest(name='name_value')
    response = client.cancel_rollout(request=request)
    print(response)