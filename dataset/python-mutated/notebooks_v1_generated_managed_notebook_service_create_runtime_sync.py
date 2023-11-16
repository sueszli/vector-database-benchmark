from google.cloud import notebooks_v1

def sample_create_runtime():
    if False:
        i = 10
        return i + 15
    client = notebooks_v1.ManagedNotebookServiceClient()
    request = notebooks_v1.CreateRuntimeRequest(parent='parent_value', runtime_id='runtime_id_value')
    operation = client.create_runtime(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)