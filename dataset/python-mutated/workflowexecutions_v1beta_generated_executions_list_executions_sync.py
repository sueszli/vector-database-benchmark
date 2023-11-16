from google.cloud.workflows import executions_v1beta

def sample_list_executions():
    if False:
        return 10
    client = executions_v1beta.ExecutionsClient()
    request = executions_v1beta.ListExecutionsRequest(parent='parent_value')
    page_result = client.list_executions(request=request)
    for response in page_result:
        print(response)