from google.cloud import workflows_v1beta

def sample_update_workflow():
    if False:
        i = 10
        return i + 15
    client = workflows_v1beta.WorkflowsClient()
    workflow = workflows_v1beta.Workflow()
    workflow.source_contents = 'source_contents_value'
    request = workflows_v1beta.UpdateWorkflowRequest(workflow=workflow)
    operation = client.update_workflow(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)