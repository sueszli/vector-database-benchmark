from google.cloud.workflows import executions_v1beta

def sample_get_execution():
    if False:
        i = 10
        return i + 15
    client = executions_v1beta.ExecutionsClient()
    request = executions_v1beta.GetExecutionRequest(name='name_value')
    response = client.get_execution(request=request)
    print(response)