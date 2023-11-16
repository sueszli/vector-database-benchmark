from google.cloud import notebooks_v2

def sample_start_instance():
    if False:
        return 10
    client = notebooks_v2.NotebookServiceClient()
    request = notebooks_v2.StartInstanceRequest(name='name_value')
    operation = client.start_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)