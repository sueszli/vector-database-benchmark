from google.cloud import notebooks_v1

def sample_switch_runtime():
    if False:
        return 10
    client = notebooks_v1.ManagedNotebookServiceClient()
    request = notebooks_v1.SwitchRuntimeRequest(name='name_value')
    operation = client.switch_runtime(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)