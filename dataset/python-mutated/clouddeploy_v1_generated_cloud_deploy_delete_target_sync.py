from google.cloud import deploy_v1

def sample_delete_target():
    if False:
        print('Hello World!')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.DeleteTargetRequest(name='name_value')
    operation = client.delete_target(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)