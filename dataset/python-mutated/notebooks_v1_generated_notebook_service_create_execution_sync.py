from google.cloud import notebooks_v1

def sample_create_execution():
    if False:
        i = 10
        return i + 15
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.CreateExecutionRequest(parent='parent_value', execution_id='execution_id_value')
    operation = client.create_execution(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)