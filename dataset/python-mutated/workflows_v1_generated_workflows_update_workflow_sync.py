from google.cloud import workflows_v1

def sample_update_workflow():
    if False:
        return 10
    client = workflows_v1.WorkflowsClient()
    workflow = workflows_v1.Workflow()
    workflow.source_contents = 'source_contents_value'
    request = workflows_v1.UpdateWorkflowRequest(workflow=workflow)
    operation = client.update_workflow(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)