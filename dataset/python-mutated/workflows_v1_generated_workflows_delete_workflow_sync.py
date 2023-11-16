from google.cloud import workflows_v1

def sample_delete_workflow():
    if False:
        return 10
    client = workflows_v1.WorkflowsClient()
    request = workflows_v1.DeleteWorkflowRequest(name='name_value')
    operation = client.delete_workflow(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)