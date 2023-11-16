from google.cloud import run_v2

def sample_delete_execution():
    if False:
        i = 10
        return i + 15
    client = run_v2.ExecutionsClient()
    request = run_v2.DeleteExecutionRequest(name='name_value')
    operation = client.delete_execution(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)