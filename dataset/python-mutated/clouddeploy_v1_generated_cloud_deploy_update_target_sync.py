from google.cloud import deploy_v1

def sample_update_target():
    if False:
        i = 10
        return i + 15
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.UpdateTargetRequest()
    operation = client.update_target(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)