from google.cloud import notebooks_v1

def sample_delete_execution():
    if False:
        return 10
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.DeleteExecutionRequest(name='name_value')
    operation = client.delete_execution(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)