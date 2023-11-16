from google.cloud import notebooks_v1beta1

def sample_delete_instance():
    if False:
        i = 10
        return i + 15
    client = notebooks_v1beta1.NotebookServiceClient()
    request = notebooks_v1beta1.DeleteInstanceRequest(name='name_value')
    operation = client.delete_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)