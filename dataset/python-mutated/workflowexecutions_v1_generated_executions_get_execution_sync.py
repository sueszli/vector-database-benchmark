from google.cloud.workflows import executions_v1

def sample_get_execution():
    if False:
        return 10
    client = executions_v1.ExecutionsClient()
    request = executions_v1.GetExecutionRequest(name='name_value')
    response = client.get_execution(request=request)
    print(response)