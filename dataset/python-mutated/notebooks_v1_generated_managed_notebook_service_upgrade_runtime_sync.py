from google.cloud import notebooks_v1

def sample_upgrade_runtime():
    if False:
        while True:
            i = 10
    client = notebooks_v1.ManagedNotebookServiceClient()
    request = notebooks_v1.UpgradeRuntimeRequest(name='name_value')
    operation = client.upgrade_runtime(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)