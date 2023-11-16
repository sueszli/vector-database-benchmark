from google.cloud import config_v1

def sample_lock_deployment():
    if False:
        i = 10
        return i + 15
    client = config_v1.ConfigClient()
    request = config_v1.LockDeploymentRequest(name='name_value')
    operation = client.lock_deployment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)