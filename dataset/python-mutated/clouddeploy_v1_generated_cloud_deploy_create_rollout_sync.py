from google.cloud import deploy_v1

def sample_create_rollout():
    if False:
        i = 10
        return i + 15
    client = deploy_v1.CloudDeployClient()
    rollout = deploy_v1.Rollout()
    rollout.target_id = 'target_id_value'
    request = deploy_v1.CreateRolloutRequest(parent='parent_value', rollout_id='rollout_id_value', rollout=rollout)
    operation = client.create_rollout(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)