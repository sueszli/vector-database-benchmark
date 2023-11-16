from google.cloud import dataproc_v1

def sample_get_workflow_template():
    if False:
        return 10
    client = dataproc_v1.WorkflowTemplateServiceClient()
    request = dataproc_v1.GetWorkflowTemplateRequest(name='name_value')
    response = client.get_workflow_template(request=request)
    print(response)