from google.cloud import config_v1

def sample_unlock_deployment():
    if False:
        for i in range(10):
            print('nop')
    client = config_v1.ConfigClient()
    request = config_v1.UnlockDeploymentRequest(name='name_value', lock_id=725)
    operation = client.unlock_deployment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)