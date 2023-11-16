from google.cloud import workflows_v1

def sample_list_workflows():
    if False:
        for i in range(10):
            print('nop')
    client = workflows_v1.WorkflowsClient()
    request = workflows_v1.ListWorkflowsRequest(parent='parent_value')
    page_result = client.list_workflows(request=request)
    for response in page_result:
        print(response)