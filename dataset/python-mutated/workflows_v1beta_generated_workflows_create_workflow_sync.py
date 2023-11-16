from google.cloud import workflows_v1beta

def sample_create_workflow():
    if False:
        return 10
    client = workflows_v1beta.WorkflowsClient()
    workflow = workflows_v1beta.Workflow()
    workflow.source_contents = 'source_contents_value'
    request = workflows_v1beta.CreateWorkflowRequest(parent='parent_value', workflow=workflow, workflow_id='workflow_id_value')
    operation = client.create_workflow(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)