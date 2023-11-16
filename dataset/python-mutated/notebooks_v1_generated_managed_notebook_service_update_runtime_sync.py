from google.cloud import notebooks_v1

def sample_update_runtime():
    if False:
        while True:
            i = 10
    client = notebooks_v1.ManagedNotebookServiceClient()
    request = notebooks_v1.UpdateRuntimeRequest()
    operation = client.update_runtime(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)