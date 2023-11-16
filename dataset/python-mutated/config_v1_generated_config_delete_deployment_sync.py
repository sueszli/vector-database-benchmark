from google.cloud import config_v1

def sample_delete_deployment():
    if False:
        while True:
            i = 10
    client = config_v1.ConfigClient()
    request = config_v1.DeleteDeploymentRequest(name='name_value')
    operation = client.delete_deployment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)