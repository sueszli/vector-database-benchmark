from google.cloud import deploy_v1

def sample_approve_rollout():
    if False:
        print('Hello World!')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.ApproveRolloutRequest(name='name_value', approved=True)
    response = client.approve_rollout(request=request)
    print(response)