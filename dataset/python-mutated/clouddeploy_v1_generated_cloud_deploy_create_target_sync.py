from google.cloud import deploy_v1

def sample_create_target():
    if False:
        return 10
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.CreateTargetRequest(parent='parent_value', target_id='target_id_value')
    operation = client.create_target(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)