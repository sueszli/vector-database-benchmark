from google.cloud import run_v2

def sample_cancel_execution():
    if False:
        for i in range(10):
            print('nop')
    client = run_v2.ExecutionsClient()
    request = run_v2.CancelExecutionRequest(name='name_value')
    operation = client.cancel_execution(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)