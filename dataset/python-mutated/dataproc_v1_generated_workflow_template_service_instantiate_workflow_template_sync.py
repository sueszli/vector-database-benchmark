from google.cloud import dataproc_v1

def sample_instantiate_workflow_template():
    if False:
        return 10
    client = dataproc_v1.WorkflowTemplateServiceClient()
    request = dataproc_v1.InstantiateWorkflowTemplateRequest(name='name_value')
    operation = client.instantiate_workflow_template(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)