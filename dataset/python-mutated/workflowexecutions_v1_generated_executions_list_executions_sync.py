from google.cloud.workflows import executions_v1

def sample_list_executions():
    if False:
        print('Hello World!')
    client = executions_v1.ExecutionsClient()
    request = executions_v1.ListExecutionsRequest(parent='parent_value')
    page_result = client.list_executions(request=request)
    for response in page_result:
        print(response)