from google.cloud.workflows import executions_v1beta

def sample_create_execution():
    if False:
        return 10
    client = executions_v1beta.ExecutionsClient()
    request = executions_v1beta.CreateExecutionRequest(parent='parent_value')
    response = client.create_execution(request=request)
    print(response)