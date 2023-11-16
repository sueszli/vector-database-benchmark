from google.cloud import deploy_v1

def sample_create_release():
    if False:
        for i in range(10):
            print('nop')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.CreateReleaseRequest(parent='parent_value', release_id='release_id_value')
    operation = client.create_release(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)