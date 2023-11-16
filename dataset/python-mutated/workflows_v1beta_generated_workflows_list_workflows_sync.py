from google.cloud import workflows_v1beta

def sample_list_workflows():
    if False:
        i = 10
        return i + 15
    client = workflows_v1beta.WorkflowsClient()
    request = workflows_v1beta.ListWorkflowsRequest(parent='parent_value')
    page_result = client.list_workflows(request=request)
    for response in page_result:
        print(response)