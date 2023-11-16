from google.cloud.workflows import executions_v1

def sample_create_execution():
    if False:
        i = 10
        return i + 15
    client = executions_v1.ExecutionsClient()
    request = executions_v1.CreateExecutionRequest(parent='parent_value')
    response = client.create_execution(request=request)
    print(response)