from google.cloud import deploy_v1

def sample_advance_rollout():
    if False:
        print('Hello World!')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.AdvanceRolloutRequest(name='name_value', phase_id='phase_id_value')
    response = client.advance_rollout(request=request)
    print(response)