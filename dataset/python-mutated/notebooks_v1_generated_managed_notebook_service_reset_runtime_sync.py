from google.cloud import notebooks_v1

def sample_reset_runtime():
    if False:
        i = 10
        return i + 15
    client = notebooks_v1.ManagedNotebookServiceClient()
    request = notebooks_v1.ResetRuntimeRequest(name='name_value')
    operation = client.reset_runtime(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)