from google.cloud.workflows import executions_v1

def sample_cancel_execution():
    if False:
        i = 10
        return i + 15
    client = executions_v1.ExecutionsClient()
    request = executions_v1.CancelExecutionRequest(name='name_value')
    response = client.cancel_execution(request=request)
    print(response)