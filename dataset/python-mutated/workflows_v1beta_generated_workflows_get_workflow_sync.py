from google.cloud import workflows_v1beta

def sample_get_workflow():
    if False:
        print('Hello World!')
    client = workflows_v1beta.WorkflowsClient()
    request = workflows_v1beta.GetWorkflowRequest(name='name_value')
    response = client.get_workflow(request=request)
    print(response)