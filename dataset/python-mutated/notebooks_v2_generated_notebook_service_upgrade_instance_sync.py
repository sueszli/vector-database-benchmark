from google.cloud import notebooks_v2

def sample_upgrade_instance():
    if False:
        return 10
    client = notebooks_v2.NotebookServiceClient()
    request = notebooks_v2.UpgradeInstanceRequest(name='name_value')
    operation = client.upgrade_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)