from google.cloud import deploy_v1

def sample_rollback_target():
    if False:
        print('Hello World!')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.RollbackTargetRequest(name='name_value', target_id='target_id_value', rollout_id='rollout_id_value')
    response = client.rollback_target(request=request)
    print(response)