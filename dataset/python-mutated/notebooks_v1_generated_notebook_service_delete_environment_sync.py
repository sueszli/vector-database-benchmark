from google.cloud import notebooks_v1

def sample_delete_environment():
    if False:
        while True:
            i = 10
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.DeleteEnvironmentRequest(name='name_value')
    operation = client.delete_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)