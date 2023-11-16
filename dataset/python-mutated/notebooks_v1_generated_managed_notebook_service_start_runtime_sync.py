from google.cloud import notebooks_v1

def sample_start_runtime():
    if False:
        print('Hello World!')
    client = notebooks_v1.ManagedNotebookServiceClient()
    request = notebooks_v1.StartRuntimeRequest(name='name_value')
    operation = client.start_runtime(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)