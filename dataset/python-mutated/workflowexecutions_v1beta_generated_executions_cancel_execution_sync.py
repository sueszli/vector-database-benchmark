from google.cloud.workflows import executions_v1beta

def sample_cancel_execution():
    if False:
        while True:
            i = 10
    client = executions_v1beta.ExecutionsClient()
    request = executions_v1beta.CancelExecutionRequest(name='name_value')
    response = client.cancel_execution(request=request)
    print(response)