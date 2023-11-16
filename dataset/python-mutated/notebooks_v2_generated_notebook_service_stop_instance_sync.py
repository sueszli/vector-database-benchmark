from google.cloud import notebooks_v2

def sample_stop_instance():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v2.NotebookServiceClient()
    request = notebooks_v2.StopInstanceRequest(name='name_value')
    operation = client.stop_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)