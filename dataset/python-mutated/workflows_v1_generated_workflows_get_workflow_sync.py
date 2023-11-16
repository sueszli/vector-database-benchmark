from google.cloud import workflows_v1

def sample_get_workflow():
    if False:
        for i in range(10):
            print('nop')
    client = workflows_v1.WorkflowsClient()
    request = workflows_v1.GetWorkflowRequest(name='name_value')
    response = client.get_workflow(request=request)
    print(response)