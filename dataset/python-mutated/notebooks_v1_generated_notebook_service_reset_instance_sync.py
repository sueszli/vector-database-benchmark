from google.cloud import notebooks_v1

def sample_reset_instance():
    if False:
        while True:
            i = 10
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.ResetInstanceRequest(name='name_value')
    operation = client.reset_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)