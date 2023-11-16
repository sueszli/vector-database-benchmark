from google.cloud import notebooks_v1

def sample_stop_runtime():
    if False:
        return 10
    client = notebooks_v1.ManagedNotebookServiceClient()
    request = notebooks_v1.StopRuntimeRequest(name='name_value')
    operation = client.stop_runtime(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)