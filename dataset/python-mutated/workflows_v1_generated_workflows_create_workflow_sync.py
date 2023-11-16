from google.cloud import workflows_v1

def sample_create_workflow():
    if False:
        for i in range(10):
            print('nop')
    client = workflows_v1.WorkflowsClient()
    workflow = workflows_v1.Workflow()
    workflow.source_contents = 'source_contents_value'
    request = workflows_v1.CreateWorkflowRequest(parent='parent_value', workflow=workflow, workflow_id='workflow_id_value')
    operation = client.create_workflow(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)