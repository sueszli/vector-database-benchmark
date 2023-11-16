from google.cloud import notebooks_v2

def sample_reset_instance():
    if False:
        i = 10
        return i + 15
    client = notebooks_v2.NotebookServiceClient()
    request = notebooks_v2.ResetInstanceRequest(name='name_value')
    operation = client.reset_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)