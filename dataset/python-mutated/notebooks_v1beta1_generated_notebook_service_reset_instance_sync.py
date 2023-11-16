from google.cloud import notebooks_v1beta1

def sample_reset_instance():
    if False:
        return 10
    client = notebooks_v1beta1.NotebookServiceClient()
    request = notebooks_v1beta1.ResetInstanceRequest(name='name_value')
    operation = client.reset_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)