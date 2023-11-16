from google.cloud import workflows_v1beta

def sample_delete_workflow():
    if False:
        while True:
            i = 10
    client = workflows_v1beta.WorkflowsClient()
    request = workflows_v1beta.DeleteWorkflowRequest(name='name_value')
    operation = client.delete_workflow(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)